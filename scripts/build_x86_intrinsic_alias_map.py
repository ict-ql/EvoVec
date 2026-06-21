#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from typing import Any


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
]

WRAPPER_RE = re.compile(r"define\b.*@(?P<name>intrinsic_wrapper_.+?)\(")
LLVM_CALL_RE = re.compile(r"@(?P<name>llvm\.[A-Za-z0-9_.]+)")


def clean_type(value: str | None) -> str:
    return (value or "").strip()


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


def element_signature(element: ET.Element) -> tuple[str, list[tuple[str, str, bool]], list[str]] | None:
    return_node = element.find("return")
    if return_node is None:
        return None
    return_type = clean_type(return_node.attrib.get("type"))
    params: list[tuple[str, str, bool]] = []
    call_args: list[str] = []
    for index, param in enumerate(element.findall("parameter")):
        ctype = clean_type(param.attrib.get("type"))
        name = clean_type(param.attrib.get("varname")) or f"arg{index}"
        if not ctype:
            return None
        is_imm = param.attrib.get("etype") == "IMM"
        params.append((ctype, name, is_imm))
        call_args.append("0" if is_imm else name)
    return return_type, params, call_args


def generate_wrapper_c(xml_path: Path, out_path: Path, limit: int = 0) -> int:
    rows: list[str] = [
        "#include <immintrin.h>",
        "#include <stdint.h>",
        "",
    ]
    count = 0
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
        rows.append(f"{return_type} {wrapper_name(intrinsic)}({param_text}) {{")
        rows.append(f"  return {intrinsic}({', '.join(call_args)});")
        rows.append("}")
        rows.append("")
        count += 1
        if limit > 0 and count >= limit:
            break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows), encoding="utf-8")
    return count


def compile_wrapper_ir(clang: str, wrapper_c: Path, out_ll: Path) -> None:
    cmd = [
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
    subprocess.run(cmd, check=True)


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
    args = parser.parse_args()

    wrapper_ir = args.wrapper_ir
    generated = 0
    if args.emit_wrapper_c:
        if args.xml is None:
            raise SystemExit("--xml is required with --emit-wrapper-c")
        generated = generate_wrapper_c(args.xml, args.emit_wrapper_c, args.limit)
    if args.compile_ir:
        if args.emit_wrapper_c is None:
            raise SystemExit("--emit-wrapper-c is required with --compile-ir")
        compile_wrapper_ir(args.clang, args.emit_wrapper_c, args.compile_ir)
        wrapper_ir = args.compile_ir
    if wrapper_ir is None:
        raise SystemExit("provide --wrapper-ir or use --emit-wrapper-c with --compile-ir")

    mapping = parse_wrapper_ir(wrapper_ir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"aliases": len(mapping), "generated_wrappers": generated, "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
