#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET


X86_FLAGS = [
    "-mssse3",
    "-msse",
    "-mavx512bw",
    "-msse3",
    "-msse4.1",
    "-msse4.2",
    "-mfma",
    "-msse2",
    "-mavx2",
    "-mavx512vnni",
    "-mavx512vl",
    "-mavx512dq",
    "-mgfni",
    "-mavx512vbmi2",
    "-mavx512vbmi",
    "-mavx512cd",
    "-mavx",
    "-mavx512f",
    "-mbmi2",
    "-madx",
]

WRAPPER_RE = re.compile(r"define\b.*@(?P<name>intrinsic_wrapper_.+?)\(")
LLVM_CALL_RE = re.compile(r"@(?P<name>llvm\.[A-Za-z0-9_.]+)")


def clean_type(value: str | None) -> str:
    ctype = (value or "").strip()
    replacements = {
        "unsigned __int8": "unsigned char",
        "unsigned __int16": "unsigned short",
        "unsigned __int32": "unsigned int",
        "unsigned __int64": "unsigned long long",
        "__int8": "signed char",
        "__int16": "short",
        "__int32": "int",
        "__int64": "long long",
    }
    for source, target in replacements.items():
        ctype = re.sub(rf"\b{re.escape(source)}\b", target, ctype)
    return ctype


def wrapper_name(intrinsic: str, imm_value: int = 0) -> str:
    return f"intrinsic_wrapper_{intrinsic}_{imm_value}"


def c_intrinsic_from_wrapper(name: str) -> str | None:
    prefix = "intrinsic_wrapper_"
    if not name.startswith(prefix):
        return None
    body = name[len(prefix) :]
    match = re.match(r"(?P<intrinsic>.+)_\d+$", body)
    if match is None:
        return None
    return match.group("intrinsic")


def intrinsic_elements(xml_path: Path) -> list[ET.Element]:
    return list(ET.parse(xml_path).getroot().findall("intrinsic"))


def immediate_value(param: ET.Element) -> str:
    name = clean_type(param.attrib.get("varname")).lower()
    immtype = clean_type(param.attrib.get("immtype")).lower()
    if name == "scale" or "index_scale" in immtype:
        return "1"
    return "0"


def element_signature(element: ET.Element) -> tuple[str, list[tuple[str, str, bool]], list[str]] | None:
    return_node = element.find("return")
    if return_node is None:
        return None
    return_type = clean_type(return_node.attrib.get("type"))
    params: list[tuple[str, str, bool]] = []
    call_args: list[str] = []
    parameter_nodes = element.findall("parameter")
    if len(parameter_nodes) == 1 and clean_type(parameter_nodes[0].attrib.get("type")) == "void":
        parameter_nodes = []
    for index, param in enumerate(parameter_nodes):
        ctype = clean_type(param.attrib.get("type"))
        name = clean_type(param.attrib.get("varname")) or f"arg{index}"
        if not ctype or ctype == "void":
            return None
        is_imm = param.attrib.get("etype") == "IMM"
        params.append((ctype, name, is_imm))
        call_args.append(immediate_value(param) if is_imm else name)
    return return_type, params, call_args


def wrapper_definitions(xml_path: Path, limit: int = 0) -> list[tuple[str, str]]:
    definitions: list[tuple[str, str]] = []
    for element in intrinsic_elements(xml_path):
        intrinsic = element.attrib.get("name", "")
        if not intrinsic or element.find("instruction") is None or element.find("operation") is None:
            continue
        signature = element_signature(element)
        if signature is None:
            continue
        return_type, params, call_args = signature
        wrapper_params = [(ctype, name) for ctype, name, is_imm in params if not is_imm]
        param_text = ", ".join(f"{ctype} {name}" for ctype, name in wrapper_params) or "void"
        function_name = wrapper_name(intrinsic, len(definitions))
        rows = [f"{return_type} {function_name}({param_text}) {{"]
        call = f"{intrinsic}({', '.join(call_args)})"
        if return_type == "void":
            rows.append(f"  {call};")
        else:
            rows.append(f"  return {call};")
        rows.append("}")
        definitions.append((intrinsic, "\n".join(rows)))
        if limit > 0 and len(definitions) >= limit:
            break
    return definitions


def render_wrapper_c(definitions: list[tuple[str, str]]) -> str:
    rows = [
        "#include <immintrin.h>",
        "#include <stdint.h>",
        "",
    ]
    for _, source in definitions:
        rows.append(source)
        rows.append("")
    return "\n".join(rows)


def generate_wrapper_c(xml_path: Path, out_path: Path, limit: int = 0) -> int:
    definitions = wrapper_definitions(xml_path, limit)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_wrapper_c(definitions), encoding="utf-8")
    return len(definitions)


def clang_command(clang: str, wrapper_c: Path, out_ll: Path) -> list[str]:
    return [
        clang,
        "-D__MM_MALLOC_H",
        "-target",
        "x86_64-unknown-none-gnu",
        "-O3",
        "-S",
        "-emit-llvm",
        *X86_FLAGS,
        str(wrapper_c),
        "-Wno-argument-outside-range",
        "-o",
        str(out_ll),
    ]


def compile_wrapper_ir(clang: str, wrapper_c: Path, out_ll: Path) -> None:
    out_ll.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(clang_command(clang, wrapper_c, out_ll), check=True)


def compile_wrapper_ir_best_effort(
    clang: str,
    definitions: list[tuple[str, str]],
    out_ll: Path,
    skipped_out: Path,
    batch_size: int,
) -> tuple[int, int]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    compiled_modules: list[str] = []
    skipped: list[dict[str, str]] = []
    compiled = 0

    with tempfile.TemporaryDirectory(prefix="x86-intrinsic-aliases-") as temporary_dir:
        temp_dir = Path(temporary_dir)
        sequence = 0

        def compile_batch(batch: list[tuple[str, str]]) -> None:
            nonlocal compiled, sequence
            sequence += 1
            source_path = temp_dir / f"batch_{sequence}.c"
            ir_path = temp_dir / f"batch_{sequence}.ll"
            source_path.write_text(render_wrapper_c(batch), encoding="utf-8")
            result = subprocess.run(
                clang_command(clang, source_path, ir_path),
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                compiled_modules.append(ir_path.read_text(encoding="utf-8", errors="replace"))
                compiled += len(batch)
                return
            if len(batch) > 1:
                middle = len(batch) // 2
                compile_batch(batch[:middle])
                compile_batch(batch[middle:])
                return
            skipped.append(
                {
                    "intrinsic": batch[0][0],
                    "error": (result.stderr or result.stdout).strip(),
                }
            )

        for start in range(0, len(definitions), batch_size):
            compile_batch(definitions[start : start + batch_size])

    out_ll.parent.mkdir(parents=True, exist_ok=True)
    out_ll.write_text("\n\n".join(compiled_modules), encoding="utf-8")
    skipped_out.parent.mkdir(parents=True, exist_ok=True)
    skipped_out.write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    return compiled, len(skipped)


def parse_wrapper_ir(ir_path: Path) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    current: str | None = None
    for line in ir_path.read_text(encoding="utf-8", errors="replace").splitlines():
        wrapper_match = WRAPPER_RE.search(line)
        if wrapper_match:
            current = c_intrinsic_from_wrapper(wrapper_match.group("name"))
            if current is not None:
                mapping.setdefault(current, [])
            continue
        if current is None:
            continue
        if line.startswith("}"):
            current = None
            continue
        for match in LLVM_CALL_RE.finditer(line):
            name = match.group("name")
            if name not in mapping[current]:
                mapping[current].append(name)
    return {key: value for key, value in mapping.items() if value}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a C-intrinsic to LLVM-intrinsic alias map using Vegen-style wrappers."
    )
    parser.add_argument("--xml", type=Path, help="Intel/Vegen data-latest.xml used to generate wrappers.")
    parser.add_argument("--emit-wrapper-c", type=Path, help="Write generated wrapper C here.")
    parser.add_argument("--compile-ir", type=Path, help="Compile generated wrapper C to textual LLVM IR here.")
    parser.add_argument("--wrapper-ir", type=Path, help="Read an existing textual LLVM IR wrapper file.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--strict", action="store_true", help="Fail if any generated wrapper does not compile.")
    parser.add_argument("--skipped-out", type=Path, help="Write wrappers skipped by best-effort compilation here.")
    args = parser.parse_args()

    wrapper_ir = args.wrapper_ir
    generated = 0
    compiled = 0
    skipped = 0
    definitions: list[tuple[str, str]] = []
    if args.emit_wrapper_c:
        if args.xml is None:
            raise SystemExit("--xml is required with --emit-wrapper-c")
        definitions = wrapper_definitions(args.xml, args.limit)
        args.emit_wrapper_c.parent.mkdir(parents=True, exist_ok=True)
        args.emit_wrapper_c.write_text(render_wrapper_c(definitions), encoding="utf-8")
        generated = len(definitions)
    if args.compile_ir:
        if args.emit_wrapper_c is None:
            raise SystemExit("--emit-wrapper-c is required with --compile-ir")
        if args.strict:
            compile_wrapper_ir(args.clang, args.emit_wrapper_c, args.compile_ir)
            compiled = generated
        else:
            skipped_out = args.skipped_out or args.out.with_suffix(".skipped.json")
            compiled, skipped = compile_wrapper_ir_best_effort(
                args.clang,
                definitions,
                args.compile_ir,
                skipped_out,
                args.batch_size,
            )
        wrapper_ir = args.compile_ir
    if wrapper_ir is None:
        raise SystemExit("provide --wrapper-ir or use --emit-wrapper-c with --compile-ir")

    mapping = parse_wrapper_ir(wrapper_ir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "aliases": len(mapping),
                "compiled_wrappers": compiled,
                "generated_wrappers": generated,
                "out": str(args.out),
                "skipped_wrappers": skipped,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
