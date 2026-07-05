from __future__ import annotations

from pathlib import Path
import re
from typing import Any


SSA_BOUNDARY = r"[A-Za-z0-9$._-]"
SSA_NAME = r"%[-a-zA-Z$._0-9]+"


def _resolve_rewrites(rewrites: dict[str, str]) -> dict[str, str]:
    def resolve(value: str) -> str:
        seen: set[str] = set()
        while value in rewrites and value not in seen:
            seen.add(value)
            value = rewrites[value]
        return value

    return {key: resolve(value) for key, value in rewrites.items()}


def _token_sub(line: str, lhs: str, rhs: str) -> str:
    pattern = re.compile(rf"(?<!{SSA_BOUNDARY}){re.escape(lhs)}(?!{SSA_BOUNDARY})")
    return pattern.sub(rhs, line)


def _apply_rewrites(ir_text: str, rewrites: dict[str, str], delete_indexes: set[int]) -> str:
    if not rewrites and not delete_indexes:
        return ir_text
    lines = ir_text.splitlines(keepends=True)
    rewrites = _resolve_rewrites(rewrites)
    for index, line in enumerate(lines):
        if index in delete_indexes:
            continue
        for lhs, rhs in rewrites.items():
            line = _token_sub(line, lhs, rhs)
        lines[index] = line
    return "".join(line for index, line in enumerate(lines) if index not in delete_indexes)


def _count_changed(before: str, after: str) -> bool:
    return before != after


BITCAST_RE = re.compile(
    rf"""(?mx)^
    \s*(?P<lhs>{SSA_NAME})\s*=\s*bitcast
    \s+(?P<src_ptr>ptr(?:\s+addrspace\(\d+\))?(?:\s+[^,%\n]+)?)
    \s+(?P<src>{SSA_NAME})
    \s+to\s+(?P<dst_ptr>ptr(?:\s+addrspace\(\d+\))?(?:\s+[^,\n]+)?)
    \s*(?:,.*)?$"""
)
ADDRSPACE_RE = re.compile(r"addrspace\((\d+)\)")


def _addrspace_of(ptr_type: str) -> str:
    match = ADDRSPACE_RE.search(ptr_type)
    return match.group(1) if match else ""


def _delete_opaque_ptr_bitcasts(ir_text: str) -> str:
    lines = ir_text.splitlines(keepends=True)
    rewrites: dict[str, str] = {}
    delete_indexes: set[int] = set()
    for index, line in enumerate(lines):
        match = BITCAST_RE.match(line)
        if not match:
            continue
        if _addrspace_of(match.group("src_ptr")) != _addrspace_of(match.group("dst_ptr")):
            continue
        rewrites[match.group("lhs")] = match.group("src")
        delete_indexes.add(index)
    return _apply_rewrites("".join(lines), rewrites, delete_indexes)


CONST_ASSIGN_RE = re.compile(
    rf"""(?mx)^
    [ \t]*(?P<lhs>{SSA_NAME})\s*=\s*
    (?P<value>
        zeroinitializer
      | <\d+\s+x\s+[^>]+>\s*<[^>]*>
      | <\d+\s+x\s+[^>]+>\s+zeroinitializer
      | <(?:i\d+|float|double|half|bfloat|ptr)\s+[^>]*>
      | (?:i\d+|float|double|half|bfloat)\s+-?(?:[\d.]+(?:[eE][+-]?\d+)?|inf|nan)
      | ptr\s+{SSA_NAME}
    )
    \s*$"""
)


def _join_multiline_constant_assignments(ir_text: str) -> str:
    lines = ir_text.splitlines(keepends=True)
    joined: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip()
        if (
            re.match(rf"{SSA_NAME}\s*=\s*<", stripped)
            and stripped.count("<") > stripped.count(">")
            and not stripped.startswith(";")
        ):
            combined = line.rstrip("\n")
            index += 1
            while index < len(lines):
                cont = lines[index].rstrip("\n")
                combined += " " + cont.strip()
                index += 1
                if combined.count(">") >= combined.count("<"):
                    break
            joined.append(combined + "\n")
        else:
            joined.append(line)
            index += 1
    return "".join(joined)


def _inline_constant_assignments(ir_text: str) -> str:
    ir_text = _join_multiline_constant_assignments(ir_text)
    lines = ir_text.splitlines(keepends=True)
    rewrites: dict[str, str] = {}
    delete_indexes: set[int] = set()
    for index, line in enumerate(lines):
        match = CONST_ASSIGN_RE.match(line)
        if not match:
            continue
        value = match.group("value").strip()
        if value.startswith("ptr "):
            value = value[4:].strip()
        elif re.match(r"^(?:i\d+|float|double|half|bfloat)\s+", value):
            value = value.split(None, 1)[1]
        rewrites[match.group("lhs")] = value
        delete_indexes.add(index)
    return _apply_rewrites("".join(lines), rewrites, delete_indexes)


BINOP_KEYWORDS = (
    "add",
    "sub",
    "mul",
    "udiv",
    "sdiv",
    "urem",
    "srem",
    "shl",
    "lshr",
    "ashr",
    "and",
    "or",
    "xor",
    "fadd",
    "fsub",
    "fmul",
    "fdiv",
    "frem",
    "icmp",
    "fcmp",
)
BINOP_ALT = "|".join(BINOP_KEYWORDS)
SCALAR_TYPES = r"(?:i\d+|float|double|half|bfloat|ptr)"

REDUNDANT_VEC_TYPE_RE = re.compile(
    rf"(?P<op>{BINOP_ALT})"
    r"(?P<flags>[^<]*?)"
    r"(?P<vecty><\d+\s+x\s+[^>]+>)"
    r"(?P<op1>\s+[^,]+,)"
    r"\s*(?P=vecty)"
    r"(?P<const>\s+<[^>]*>)"
)
REDUNDANT_VEC_TYPE_FIRST_RE = re.compile(
    rf"(?P<op>{BINOP_ALT})"
    r"(?P<flags>[^<]*?)"
    r"(?P<vecty><\d+\s+x\s+[^>]+>)"
    r"\s+(?P=vecty)"
    r"(?P<const>\s+<[^>]*>)"
)
REDUNDANT_SCALAR_TYPE_RE = re.compile(
    rf"(?P<op>{BINOP_ALT})"
    r"(?P<flags>\s+(?:nsw|nuw|exact|eq|ne|ugt|uge|ult|ule|sgt|sge|slt|sle"
    r"|oeq|ogt|oge|olt|ole|one|ord|ueq|une|uno|true|false)\s+|\s+)"
    rf"(?P<sty>{SCALAR_TYPES})"
    r"\s+(?P=sty)"
    r"(?P<rest>\s+.+)"
)


def _strip_redundant_operand_types(ir_text: str) -> str:
    fixed_lines: list[str] = []
    for line in ir_text.splitlines(keepends=True):
        line = REDUNDANT_VEC_TYPE_RE.sub(
            lambda m: m.group("op") + m.group("flags") + m.group("vecty") + m.group("op1") + m.group("const"),
            line,
        )
        line = REDUNDANT_VEC_TYPE_FIRST_RE.sub(
            lambda m: m.group("op") + m.group("flags") + m.group("vecty") + m.group("const"),
            line,
        )
        line = REDUNDANT_SCALAR_TYPE_RE.sub(
            lambda m: m.group("op") + m.group("flags") + m.group("sty") + m.group("rest"),
            line,
        )
        fixed_lines.append(line)
    return "".join(fixed_lines)


VECTOR_CONSTANT_RE = re.compile(r"<\d+\s+x\s+(?P<elem>\w+)>\s*<(?P<body>[^>]+)>")
INNER_VECTOR_CONSTANT_RE = re.compile(
    r"<(?P<elem>i\d+|float|double|half|bfloat)\s+(?P<body>[^<>]*,[^<>]*)>"
)


def _fix_vector_constant_elements(ir_text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        elem_type = match.group("elem")
        body = match.group("body")
        fixed_parts: list[str] = []
        for part in body.split(","):
            item = part.strip()
            if not item:
                continue
            if (
                item.startswith(elem_type + " ")
                or item in {"undef", "poison", "zeroinitializer", "true", "false"}
                or re.match(r"^[if]\d+\s", item)
            ):
                fixed_parts.append(item)
            elif re.match(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$", item):
                fixed_parts.append(f"{elem_type} {item}")
            else:
                fixed_parts.append(item)
        return f"<{match.group(0).split('<', 1)[1].split('>', 1)[0]}> <{', '.join(fixed_parts)}>"

    return VECTOR_CONSTANT_RE.sub(repl, ir_text)


def _fix_inner_vector_constant_elements(ir_text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        elem_type = match.group("elem")
        body = match.group("body")
        parts = [elem_type + " " + body.split(",", 1)[0].strip()]
        for part in body.split(",")[1:]:
            item = part.strip()
            if not item:
                continue
            if item.startswith(elem_type + " ") or item in {"undef", "poison", "zeroinitializer", "true", "false"}:
                parts.append(item)
            elif re.match(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$", item):
                parts.append(f"{elem_type} {item}")
            else:
                parts.append(item)
        return f"<{', '.join(parts)}>"

    return INNER_VECTOR_CONSTANT_RE.sub(repl, ir_text)


ATTR_GROUP_RE = re.compile(r"^\s*attributes\s+#\d+\s*=\s*\{.*\}\s*$")
ATTR_COLON_RE = re.compile(r'"(?P<key>[-A-Za-z0-9_]+)"\s*:\s*"(?P<value>[^"]*)"')


def _fix_attribute_key_value_colons(ir_text: str) -> str:
    lines: list[str] = []
    for line in ir_text.splitlines(keepends=True):
        if ATTR_GROUP_RE.match(line):
            line = ATTR_COLON_RE.sub(lambda m: f'"{m.group("key")}"="{m.group("value")}"', line)
        lines.append(line)
    return "".join(lines)


INLINE_GEP_ADD_CALL_RE = re.compile(
    rf"(?P<ty>i\d+)\s+add\s+\(\s*(?P=ty)\s+(?P<lhs>{SSA_NAME}|-?\d+)\s*,\s*(?:(?P=ty)\s+)?(?P<rhs>{SSA_NAME}|-?\d+)\s*\)"
)
INLINE_GEP_PAREN_ADD_RE = re.compile(
    rf"(?P<ty>i\d+)\s+\(\s*add\s+\(\s*(?P=ty)\s+(?P<lhs>{SSA_NAME}|-?\d+)\s*\)\s*,\s*(?:(?P=ty)\s+)?(?P<rhs>{SSA_NAME}|-?\d+)\s*\)"
)
INLINE_CONSTEXPR_RE = re.compile(
    r"(?P<ty>i\d+|float|double|ptr)\s+\("
    r"(?P<op>add|sub|mul|shl|lshr|ashr|and|or|xor|zext|sext|trunc|bitcast)"
    r"\s+(?P<body>[^()]+)\)"
)


def _fix_inline_gep_add_exprs(ir_text: str) -> str:
    lines = ir_text.splitlines(keepends=True)
    counter = 0
    fixed: list[str] = []
    for line in lines:
        match = INLINE_GEP_ADD_CALL_RE.search(line) or INLINE_GEP_PAREN_ADD_RE.search(line)
        if not match or line.lstrip().startswith(";"):
            fixed.append(line)
            continue
        tmp = f"%_pp_gepadd_{counter}"
        counter += 1
        ty = match.group("ty")
        extracted = f"  {tmp} = add {ty} {match.group('lhs')}, {match.group('rhs')}\n"
        fixed.append(extracted)
        fixed.append(line[: match.start()] + f"{ty} {tmp}" + line[match.end() :])
    return "".join(fixed)


def _fix_inline_constexprs(ir_text: str) -> str:
    lines = ir_text.splitlines(keepends=True)
    counter = 0
    changed = True
    while changed and counter < 100:
        changed = False
        next_lines: list[str] = []
        for line in lines:
            match = INLINE_CONSTEXPR_RE.search(line)
            if not match or line.lstrip().startswith(";"):
                next_lines.append(line)
                continue
            tmp = f"%_pp_fixup_{counter}"
            counter += 1
            op = match.group("op")
            ty = match.group("ty")
            body = match.group("body").strip()
            if op in {"zext", "sext", "trunc", "bitcast"}:
                extracted = f"  {tmp} = {op} {body}\n"
            else:
                extracted = f"  {tmp} = {op} {ty} {body}\n"
            replacement = f"{ty} {tmp}"
            next_lines.append(extracted)
            next_lines.append(line[: match.start()] + replacement + line[match.end() :])
            changed = True
        lines = next_lines
    return "".join(lines)


def postprocess_ir_text(ir_text: str) -> tuple[str, dict[str, Any]]:
    transforms = [
        ("attribute_key_value_colons", _fix_attribute_key_value_colons),
        ("opaque_ptr_bitcasts", _delete_opaque_ptr_bitcasts),
        ("constant_assignments", _inline_constant_assignments),
        ("redundant_operand_types", _strip_redundant_operand_types),
        ("vector_constant_elements", _fix_vector_constant_elements),
        ("inner_vector_constant_elements", _fix_inner_vector_constant_elements),
        ("inline_gep_add_exprs", _fix_inline_gep_add_exprs),
        ("inline_constexprs", _fix_inline_constexprs),
    ]
    summary: dict[str, Any] = {"changed": False, "transforms": []}
    current = ir_text
    for name, transform in transforms:
        before = current
        current = transform(current)
        if _count_changed(before, current):
            summary["changed"] = True
            summary["transforms"].append(name)
    return current, summary


def postprocess_ir_file(path: Path) -> dict[str, Any]:
    original = path.read_text(encoding="utf-8", errors="replace")
    processed, summary = postprocess_ir_text(original)
    if processed != original:
        path.write_text(processed, encoding="utf-8")
    summary["path"] = str(path)
    return summary
