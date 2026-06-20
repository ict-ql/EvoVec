#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from typing import Any


LANE_SUFFIXES = {
    "B": "b",
    "W": "w",
    "D": "d",
    "Q": "q",
}


def clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


def raw_text(value: str | None) -> str:
    return (value or "").strip()


def normalize_extension(value: str) -> str:
    return value.upper().replace("-", "_").replace(".", "_")


def extension_from_intrinsic(element: ET.Element) -> str:
    cpuid_values = [normalize_extension(child.text or "") for child in element.findall("CPUID") if child.text]
    if cpuid_values:
        avx512 = [item for item in cpuid_values if item.startswith("AVX512")]
        if avx512:
            return avx512[0]
        avx = [item for item in cpuid_values if item.startswith("AVX")]
        if avx:
            return avx[0]
        sse = [item for item in cpuid_values if item.startswith("SSE") or item == "SSSE3"]
        if sse:
            return sse[0]
        amx = [item for item in cpuid_values if item.startswith("AMX")]
        if amx:
            return amx[0]
        return cpuid_values[0]
    return normalize_extension(element.attrib.get("tech", "x86"))


def operation_lines(element: ET.Element) -> list[str]:
    operation = element.find("operation")
    if operation is None:
        return []
    return [line.strip() for line in raw_text(operation.text).splitlines() if line.strip()]


def instruction_texts(element: ET.Element) -> list[str]:
    texts: list[str] = []
    for instruction in element.findall("instruction"):
        name = instruction.attrib.get("name", "")
        form = instruction.attrib.get("form", "")
        text = clean_text(f"{name} {form}")
        if text:
            texts.append(text)
    return texts


def primary_instruction(element: ET.Element) -> str:
    instructions = instruction_texts(element)
    return instructions[0] if instructions else ""


def operand_role(tag: str) -> str:
    return "dst" if tag == "return" else "src"


def operands(element: ET.Element) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for child in list(element):
        if child.tag not in {"return", "parameter"}:
            continue
        name = child.attrib.get("varname", "dst" if child.tag == "return" else "")
        if not name:
            continue
        operand = {
            "name": name,
            "role": operand_role(child.tag),
            "type": child.attrib.get("etype") or child.attrib.get("type", ""),
        }
        if child.attrib.get("memwidth"):
            operand["memwidth"] = child.attrib["memwidth"]
        items.append(operand)
    return items


def llvm_op_from_mnemonic(mnemonic: str) -> str:
    lowered = mnemonic.lower()
    if len(mnemonic) > 1 and mnemonic[-1].upper() in LANE_SUFFIXES:
        return lowered[:-1] + "." + LANE_SUFFIXES[mnemonic[-1].upper()]
    return lowered


def llvm_extension_prefix(extension: str) -> str:
    normalized = normalize_extension(extension)
    if normalized.startswith("AVX512"):
        return "avx512"
    return normalized.lower().replace("_", ".")


def llvm_aliases(extension: str, instructions: list[str]) -> list[str]:
    aliases: list[str] = []
    prefix = llvm_extension_prefix(extension)
    for instruction in instructions:
        mnemonic = instruction.split()[0].upper()
        op = llvm_op_from_mnemonic(mnemonic)
        aliases.append(f"llvm.x86.{prefix}.{op}")
        if op.startswith("v") and len(op) > 1:
            aliases.append(f"llvm.x86.{prefix}.{op[1:]}")
        if prefix == "avx512":
            aliases.append(f"llvm.x86.avx512.{op}.512")
            if op.startswith("v") and len(op) > 1:
                aliases.append(f"llvm.x86.avx512.{op[1:]}.512")
    return aliases


def spec_from_intrinsic(element: ET.Element) -> dict[str, Any]:
    intrinsic = element.attrib.get("name", "")
    extension = extension_from_intrinsic(element)
    instructions = instruction_texts(element)
    cpuid = [clean_text(child.text) for child in element.findall("CPUID") if clean_text(child.text)]
    aliases = [intrinsic, *instructions, *[text.split()[0].lower() for text in instructions], *llvm_aliases(extension, instructions)]
    label = aliases[-1] if aliases else f"x86.{intrinsic.lstrip('_')}"
    return {
        "label": label,
        "architecture": "x86",
        "extension": extension,
        "intrinsic": intrinsic,
        "instruction": primary_instruction(element),
        "aliases": list(dict.fromkeys(alias for alias in aliases if alias)),
        "cpuid": cpuid,
        "category": clean_text(element.findtext("category")),
        "description": clean_text(element.findtext("description")),
        "header": clean_text(element.findtext("header")),
        "operands": operands(element),
        "operation": operation_lines(element),
        "source": "intel_intrinsics_xml",
    }


def include_record(record: dict[str, Any], include_extensions: set[str], exclude_extensions: set[str]) -> bool:
    extension = normalize_extension(str(record.get("extension", "")))
    if include_extensions and extension not in include_extensions:
        return False
    if exclude_extensions and extension in exclude_extensions:
        return False
    return bool(record.get("operation")) and bool(record.get("instruction"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def parse_extensions(values: list[str] | None) -> set[str]:
    extensions: set[str] = set()
    for value in values or []:
        for item in re.split(r"[, ]+", value):
            if item.strip():
                extensions.add(normalize_extension(item.strip()))
    return extensions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Intel/Vegen intrinsics XML into instruction micro-op spec JSONL."
    )
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-extension", action="append")
    parser.add_argument("--exclude-extension", action="append")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    include_extensions = parse_extensions(args.include_extension)
    exclude_extensions = parse_extensions(args.exclude_extension)
    root = ET.parse(args.xml).getroot()
    rows: list[dict[str, Any]] = []
    extension_counts: Counter[str] = Counter()
    for element in root.findall("intrinsic"):
        record = spec_from_intrinsic(element)
        if not include_record(record, include_extensions, exclude_extensions):
            continue
        rows.append(record)
        extension_counts[str(record["extension"])] += 1
        if args.limit > 0 and len(rows) >= args.limit:
            break

    write_jsonl(args.out, rows)
    summary = {
        "input": str(args.xml),
        "out": str(args.out),
        "rows": len(rows),
        "extensions": extension_counts.most_common(),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
