#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any


MNEMONIC_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\b")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run_cmd(cmd: list[str], timeout: int) -> dict[str, Any]:
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_sec": round(time.time() - start, 3),
            "ok": proc.returncode == 0,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "elapsed_sec": round(time.time() - start, 3),
            "ok": False,
            "error": "timeout",
        }


def strip_asm_comment(line: str) -> str:
    if "#" in line:
        line = line.split("#", 1)[0]
    return line.strip()


def extract_instruction_counts(asm_text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for raw_line in asm_text.splitlines():
        line = strip_asm_comment(raw_line)
        if not line:
            continue
        if line.startswith("."):
            continue
        if line.endswith(":"):
            continue
        match = MNEMONIC_RE.match(line)
        if not match:
            continue
        mnemonic = match.group(1).lower()
        if mnemonic in {"end", "type", "size"}:
            continue
        counts[mnemonic] += 1
    return counts


def record_ir(row: dict[str, Any], field: str) -> str | None:
    value = row.get(field)
    if value is None and isinstance(row.get("ir"), dict):
        value = row["ir"].get(field)
    return str(value) if value else None


def split_inputs(args: argparse.Namespace) -> list[Path]:
    if args.split_jsonl:
        return args.split_jsonl
    if not args.split_dir:
        raise SystemExit("provide --split-dir or one or more --split-jsonl paths")
    paths = [args.split_dir / f"{split}.jsonl" for split in ("train", "val", "test")]
    return [path for path in paths if path.exists()]


def split_name(path: Path) -> str:
    name = path.name
    if name.endswith(".jsonl"):
        return name[:-6]
    return path.stem


def compile_ir_to_asm(
    llc: str,
    ir_text: str,
    asm_path: Path,
    target: str,
    mcpu: str,
    opt_level: str,
    timeout: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temp_dir:
        ir_path = Path(temp_dir) / "input.ll"
        ir_path.write_text(ir_text, encoding="utf-8")
        cmd = [
            llc,
            opt_level,
            "-mtriple",
            target,
            "-mcpu",
            mcpu,
            "-filetype=asm",
            "-o",
            str(asm_path),
            str(ir_path),
        ]
        return run_cmd(cmd, timeout=timeout)


def compile_ir_to_asm_llvmlite(
    ir_text: str,
    target: str,
    mcpu: str,
    features: str,
) -> dict[str, Any]:
    start = time.time()
    try:
        llvm = importlib.import_module("llvmlite.binding")
        llvm.initialize_all_targets()
        llvm.initialize_all_asmprinters()
        module = llvm.parse_assembly(ir_text)
        module.verify()
        llvm_target = llvm.Target.from_triple(target)
        target_machine = llvm_target.create_target_machine(cpu=mcpu, features=features)
        asm_text = target_machine.emit_assembly(module)
        return {
            "cmd": ["llvmlite.binding.TargetMachine.emit_assembly"],
            "returncode": 0,
            "stdout": asm_text,
            "stderr": "",
            "elapsed_sec": round(time.time() - start, 3),
            "ok": True,
        }
    except Exception as exc:
        return {
            "cmd": ["llvmlite.binding.TargetMachine.emit_assembly"],
            "returncode": 1,
            "stdout": "",
            "stderr": str(exc),
            "elapsed_sec": round(time.time() - start, 3),
            "ok": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile IR-OptSet split records with llc and extract x86 instruction labels."
    )
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--split-jsonl", type=Path, nargs="*")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/iroptset/instruction_labels/pilot_10k_v1"))
    parser.add_argument("--compile-ir-field", default="o3_ir")
    parser.add_argument("--backend", choices=["auto", "llc", "llvmlite"], default="auto")
    parser.add_argument("--llc", default="llc")
    parser.add_argument("--target", default="x86_64-pc-linux-gnu")
    parser.add_argument("--mcpu", default="native")
    parser.add_argument("--mattr", default="", help="LLVM feature string for llvmlite backend.")
    parser.add_argument("--opt-level", default="-O3")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--limit", type=int, default=0, help="Optional per-split row limit; 0 means all rows.")
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()

    llc_path = shutil.which(args.llc)
    backend = args.backend
    if backend == "auto":
        backend = "llc" if llc_path else "llvmlite"
    if backend == "llc" and not llc_path:
        raise SystemExit(
            f"could not find llc executable {args.llc!r}; install LLVM tools or pass --llc /path/to/llc"
        )
    if backend == "llvmlite" and importlib.util.find_spec("llvmlite.binding") is None:
        raise SystemExit(
            "could not import llvmlite.binding; install llvmlite or use --backend llc"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for input_path in split_inputs(args):
        split = split_name(input_path)
        rows = load_jsonl(input_path)
        if args.limit > 0:
            rows = rows[: args.limit]
        labels_path = args.out_dir / f"{split}.labels.jsonl"
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        ok = 0
        failed = 0
        instruction_vocab: Counter[str] = Counter()
        with labels_path.open("w", encoding="utf-8") as labels_file:
            for ordinal, row in enumerate(rows):
                ir_text = record_ir(row, args.compile_ir_field)
                record_id = row.get("dataset_index", ordinal)
                artifact_dir = args.out_dir / "artifacts" / split / f"{int(record_id):08d}"
                asm_path = artifact_dir / "output.s"
                if ir_text is None:
                    failed += 1
                    labels_file.write(
                        json.dumps(
                            {
                                "ok": False,
                                "error": f"missing IR field {args.compile_ir_field}",
                                "dataset_index": row.get("dataset_index"),
                                "split": split,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    continue

                if backend == "llc":
                    assert llc_path is not None
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    result = compile_ir_to_asm(
                        llc_path,
                        ir_text,
                        asm_path,
                        args.target,
                        args.mcpu,
                        args.opt_level,
                        args.timeout,
                    )
                    asm_text = asm_path.read_text(encoding="utf-8", errors="replace") if result["ok"] else ""
                else:
                    result = compile_ir_to_asm_llvmlite(
                        ir_text,
                        args.target,
                        args.mcpu,
                        args.mattr,
                    )
                    asm_text = str(result["stdout"]) if result["ok"] else ""
                    if result["ok"] and args.keep_artifacts:
                        artifact_dir.mkdir(parents=True, exist_ok=True)
                        asm_path.write_text(asm_text, encoding="utf-8")
                if not result["ok"]:
                    failed += 1
                    label_record = {
                        "ok": False,
                        "split": split,
                        "dataset_index": row.get("dataset_index"),
                        "repo_name": row.get("repo_name"),
                        "repo_file_path": row.get("repo_file_path"),
                        "function_name": row.get("function_name"),
                        "structural_hash": row.get("structural_hash"),
                        "compile": result,
                    }
                    labels_file.write(json.dumps(label_record, sort_keys=True) + "\n")
                    continue

                counts = extract_instruction_counts(asm_text)
                instruction_vocab.update(counts)
                ok += 1
                label_record = {
                    "ok": True,
                    "split": split,
                    "dataset_index": row.get("dataset_index"),
                    "repo_name": row.get("repo_name"),
                    "repo_file_path": row.get("repo_file_path"),
                    "function_name": row.get("function_name"),
                    "structural_hash": row.get("structural_hash"),
                    "compile_ir_field": args.compile_ir_field,
                    "target": args.target,
                    "mcpu": args.mcpu,
                    "mattr": args.mattr,
                    "opt_level": args.opt_level,
                    "backend": backend,
                    "instruction_counts": dict(sorted(counts.items())),
                    "instructions": sorted(counts),
                    "instruction_total": sum(counts.values()),
                    "instruction_unique": len(counts),
                    "asm_path": str(asm_path) if args.keep_artifacts else None,
                    "compile": {
                        "cmd": result["cmd"],
                        "returncode": result["returncode"],
                        "elapsed_sec": result["elapsed_sec"],
                        "ok": result["ok"],
                    },
                }
                labels_file.write(json.dumps(label_record, sort_keys=True) + "\n")
                if not args.keep_artifacts:
                    asm_path.unlink(missing_ok=True)
                    try:
                        artifact_dir.rmdir()
                    except OSError:
                        pass
        summaries[split] = {
            "input": str(input_path),
            "labels": str(labels_path),
            "rows": len(rows),
            "ok": ok,
            "failed": failed,
            "instruction_vocab_size": len(instruction_vocab),
            "top_instructions": instruction_vocab.most_common(50),
        }
        print(json.dumps({"split": split, **summaries[split]}, sort_keys=True))

    summary = {
        "backend": backend,
        "llc": llc_path,
        "compile_ir_field": args.compile_ir_field,
        "target": args.target,
        "mcpu": args.mcpu,
        "mattr": args.mattr,
        "opt_level": args.opt_level,
        "splits": summaries,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if all(item["failed"] == 0 for item in summaries.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
