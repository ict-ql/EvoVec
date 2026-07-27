#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"expected object row in {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def run_cmd(cmd: list[str], *, timeout: int | None = None) -> dict[str, Any]:
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
    except FileNotFoundError as exc:
        return {
            "cmd": cmd,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
            "elapsed_sec": round(time.time() - start, 3),
            "ok": False,
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


def parse_tsvc_output(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return {"ok": False, "error": "empty stdout"}
    data_line = lines[-1]
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", data_line)
    if len(numbers) < 2:
        return {"ok": False, "error": f"could not parse TSVC line: {data_line!r}"}
    try:
        return {
            "ok": True,
            "line": data_line,
            "time_sec": float(numbers[-2]),
            "checksum": float(numbers[-1]),
            "stdout": stdout,
        }
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "line": data_line}


def resolve_task_path(task: dict[str, Any], key: str, fallback: Path) -> Path:
    text = str(task.get(key) or "").strip()
    return Path(text) if text else fallback


def resolve_tsvc_root(args: argparse.Namespace, task: dict[str, Any]) -> Path:
    text = str(task.get("root") or "").strip()
    if text:
        return Path(text)
    return args.tsvc_root


def task_paths(args: argparse.Namespace, task: dict[str, Any]) -> dict[str, Path]:
    kernel = str(task["kernel"])
    root = resolve_tsvc_root(args, task)
    kernel_dir = root / "kernels" / kernel
    return {
        "support_dir": resolve_task_path(task, "support_dir", root / "support"),
        "reference_kernel": resolve_task_path(task, "reference_kernel", kernel_dir / "reference_kernel.c"),
        "harness": resolve_task_path(task, "harness", kernel_dir / "harness.c"),
    }


def parse_compiler_spec(spec: str) -> tuple[str, list[str]]:
    posix = os.name != "nt"
    if "=" not in spec:
        cmd = shlex.split(spec, posix=posix)
        label = Path(cmd[0]).name
        return label, cmd
    label, command = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise SystemExit(f"empty compiler label in {spec!r}")
    cmd = shlex.split(command.strip(), posix=posix)
    if not cmd:
        raise SystemExit(f"empty compiler command in {spec!r}")
    return label, cmd


def build_and_run(
    args: argparse.Namespace,
    task: dict[str, Any],
    label: str,
    compiler_cmd: list[str],
    task_dir: Path,
    expected_checksum: float | None,
) -> dict[str, Any]:
    paths = task_paths(args, task)
    support = paths["support_dir"]
    out_dir = task_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = out_dir / "reference.exe"
    cmd = [
        *compiler_cmd,
        f"-O{args.opt_level}",
        f"-march={args.march}",
        "-no-pie",
        "-I",
        str(support),
        *args.extra_arg,
        str(paths["reference_kernel"]),
        str(support / "common.c"),
        str(support / "dummy.c"),
        str(support / "globals.c"),
        str(paths["harness"]),
        "-lm",
        "-o",
        str(exe),
    ]
    compile_result = run_cmd(cmd, timeout=args.build_timeout)
    run_result: dict[str, Any] | None = None
    parsed: dict[str, Any] | None = None
    checksum_ok: bool | None = None
    if compile_result["ok"]:
        run_result = run_cmd([str(exe)], timeout=args.run_timeout)
        parsed = parse_tsvc_output(str(run_result["stdout"])) if run_result["ok"] else None
        if expected_checksum is not None and parsed and parsed["ok"]:
            checksum_ok = math.isclose(
                float(parsed["checksum"]),
                expected_checksum,
                rel_tol=0.0,
                abs_tol=args.checksum_atol,
            )
    ok = bool(compile_result["ok"] and run_result and run_result["ok"] and parsed and parsed["ok"])
    if checksum_ok is not None:
        ok = ok and checksum_ok
    record = {
        "compiler": label,
        "exe": str(exe),
        "compile": compile_result,
        "run": run_result,
        "parsed": parsed,
        "expected_checksum": expected_checksum,
        "checksum_ok": checksum_ok,
        "ok": ok,
    }
    (out_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def run_task(args: argparse.Namespace, task: dict[str, Any], compilers: dict[str, list[str]]) -> dict[str, Any]:
    kernel = str(task["kernel"])
    task_dir = args.out_dir / kernel
    task_dir.mkdir(parents=True, exist_ok=True)

    reference_label = args.reference
    if reference_label not in compilers:
        raise SystemExit(f"--reference {reference_label!r} is not in --compiler specs")

    reference = build_and_run(args, task, reference_label, compilers[reference_label], task_dir, None)
    expected_checksum = None
    reference_time = None
    if reference["ok"]:
        parsed = reference["parsed"] or {}
        expected_checksum = float(parsed["checksum"])
        reference_time = float(parsed["time_sec"])

    compiler_records: dict[str, Any] = {reference_label: reference}
    for label, compiler_cmd in compilers.items():
        if label == reference_label:
            continue
        compiler_records[label] = build_and_run(
            args,
            task,
            label,
            compiler_cmd,
            task_dir,
            expected_checksum,
        )

    speedups: dict[str, float | None] = {}
    for label, record in compiler_records.items():
        parsed = record.get("parsed") or {}
        time_sec = parsed.get("time_sec")
        if label == reference_label:
            continue
        speedups[f"{label}_vs_{reference_label}"] = (
            float(reference_time) / float(time_sec)
            if reference_time and time_sec and float(time_sec) > 0.0
            else None
        )

    record = {
        "task_id": task.get("task_id"),
        "kernel": kernel,
        "reference": reference_label,
        "reference_time_sec": reference_time,
        "expected_checksum": expected_checksum,
        "compilers": compiler_records,
        "speedups": speedups,
    }
    (task_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def geomean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0.0 and math.isfinite(value)]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def summarize(records: list[dict[str, Any]], compiler_labels: list[str], reference_label: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "tasks": len(records),
        "reference": reference_label,
        "compiler_labels": compiler_labels,
    }
    for label in compiler_labels:
        ok_records = [
            record
            for record in records
            if record.get("compilers", {}).get(label, {}).get("ok")
        ]
        times = [
            float(record["compilers"][label]["parsed"]["time_sec"])
            for record in ok_records
            if record["compilers"][label].get("parsed")
            and record["compilers"][label]["parsed"].get("time_sec") is not None
        ]
        summary[f"{label}_ok"] = len(ok_records)
        summary[f"{label}_mean_time_sec"] = sum(times) / len(times) if times else None
    for label in compiler_labels:
        if label == reference_label:
            continue
        key = f"{label}_vs_{reference_label}"
        values = [
            float(record.get("speedups", {}).get(key))
            for record in records
            if record.get("speedups", {}).get(key) is not None
        ]
        summary[f"{key}_mean"] = sum(values) / len(values) if values else None
        summary[f"{key}_geomean"] = geomean(values)
        summary[f"{key}_count"] = len(values)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TSVC compiler baselines and report speedups versus a reference compiler.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tsvc-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--compiler",
        action="append",
        default=[],
        help="Compiler spec, either label=command or command. Repeatable, e.g. --compiler gcc=gcc --compiler llvm19=/opt/llvm-19.1.0/bin/clang --compiler icx=icx.",
    )
    parser.add_argument("--reference", default="gcc", help="Compiler label used as checksum/time reference.")
    parser.add_argument("--march", default="native")
    parser.add_argument("--opt-level", default="3", choices=["0", "1", "2", "3", "s", "z"])
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--build-timeout", type=int, default=120)
    parser.add_argument("--run-timeout", type=int, default=300)
    parser.add_argument("--checksum-atol", type=float, default=1e-3)
    args = parser.parse_args()

    compiler_specs = args.compiler or ["gcc=gcc", "llvm19=clang", "icx=icx"]
    compilers = dict(parse_compiler_spec(spec) for spec in compiler_specs)
    if args.reference not in compilers:
        raise SystemExit(f"--reference {args.reference!r} is not in --compiler specs")

    tasks = load_jsonl(args.manifest)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    if args.jobs <= 1:
        for task in tasks:
            records.append(run_task(args, task, compilers))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_task, args, task, compilers) for task in tasks]
            for future in as_completed(futures):
                records.append(future.result())
        records.sort(key=lambda row: str(row.get("kernel", "")))

    records_path = args.out_dir / "records.jsonl"
    summary_path = args.out_dir / "summary.json"
    write_jsonl(records_path, records)
    summary = summarize(records, list(compilers), args.reference)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"records": str(records_path), "summary": str(summary_path), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
