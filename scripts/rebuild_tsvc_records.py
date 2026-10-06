#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_DIR_PATTERN = re.compile(r"^(\d+)_([^/]+)$")
ITER_DIR_PATTERN = re.compile(r"^iter_(\d+)$")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def backup_existing(path: Path, timestamp: str) -> Path | None:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.bak.{timestamp}")
    shutil.copy2(path, backup)
    return backup


def validation_status(result: dict[str, Any], *, alive2: bool) -> str:
    status = result.get("status")
    if isinstance(status, str):
        return status
    if result.get("ok") or (alive2 and result.get("verified")):
        return "pass"
    if result.get("counterexample"):
        return "fail"
    stdout = str(result.get("stdout", ""))
    if alive2 and "failed-to-prove transformations" in stdout:
        return "inconclusive"
    if result.get("enabled") is False:
        return "disabled"
    return "fail"


def attempt_directories(mode_dir: Path) -> list[tuple[int, Path]]:
    attempts = [(0, mode_dir)]
    for path in mode_dir.iterdir():
        if not path.is_dir():
            continue
        match = ITER_DIR_PATTERN.fullmatch(path.name)
        if match:
            attempts.append((int(match.group(1)), path))
    return sorted(attempts)


def rebuild_attempt(
    attempt_dir: Path,
    iteration: int,
    reference_time: float | None,
    scalar_time: float | None,
) -> dict[str, Any] | None:
    verify_path = attempt_dir / "verify.json"
    llm_path = attempt_dir / "llm.json"
    if not verify_path.exists() and not llm_path.exists():
        return None

    verify = read_json(verify_path)
    llm = read_json(llm_path)
    differential = verify.get("differential") if isinstance(verify.get("differential"), dict) else {}
    parsed = differential.get("parsed") if isinstance(differential.get("parsed"), dict) else {}
    compile_result = differential.get("compile") if isinstance(differential.get("compile"), dict) else {}
    run_result = differential.get("run") if isinstance(differential.get("run"), dict) else {}
    alive2_result = verify.get("alive2") if isinstance(verify.get("alive2"), dict) else {}
    fuzz_result = (
        verify.get("differential_fuzz") if isinstance(verify.get("differential_fuzz"), dict) else {}
    )
    alive2_status = validation_status(alive2_result, alive2=True)
    fuzz_status = validation_status(fuzz_result, alive2=False)
    observed_time = parsed.get("time_sec")
    attempt: dict[str, Any] = {
        "iteration": iteration,
        "candidate_origin": "seed" if iteration == 0 else "generated",
        "candidate_ir": str(attempt_dir / "candidate.ll"),
        "prompt_path": str(attempt_dir / "prompt.txt"),
        "llm_path": str(llm_path),
        "postprocess_path": str(attempt_dir / "postprocess.json"),
        "verify_path": str(verify_path),
        "llm_ok": bool(llm.get("ok")),
        "llm_elapsed_sec": llm.get("elapsed_sec"),
        "skipped_existing_generation": True,
        "ok": bool(verify.get("ok")),
        "repairable": bool(verify.get("repairable", not verify.get("ok"))),
        "semantic_statuses": verify.get("semantic_statuses", [alive2_status, fuzz_status]),
        "alive2_ok": alive2_status == "pass",
        "alive2_status": alive2_status,
        "differential_fuzz_ok": fuzz_status == "pass",
        "differential_fuzz_status": fuzz_status,
        "tsvc_ok": bool(differential.get("ok")),
        "compile_ok": bool(compile_result.get("ok")),
        "run_ok": bool(run_result.get("ok")),
        "checksum_ok": bool(differential.get("checksum_ok")),
        "observed_checksum": parsed.get("checksum"),
        "observed_time_sec": observed_time,
        "reference_time_sec": reference_time,
        "scalar_ir_time_sec": scalar_time,
        "speedup_vs_scalar_ir": (
            float(scalar_time) / float(observed_time)
            if scalar_time and observed_time and float(observed_time) > 0.0
            else None
        ),
        "speedup_vs_reference": (
            float(reference_time) / float(observed_time)
            if reference_time and observed_time and float(observed_time) > 0.0
            else None
        ),
    }
    if not verify and llm:
        attempt["error"] = llm.get("error") or llm.get("stderr") or "missing verify.json"
    return attempt


def rebuild_mode(
    mode_dir: Path,
    mode: str,
    reference_time: float | None,
    scalar_time: float | None,
) -> dict[str, Any] | None:
    attempts = [
        attempt
        for iteration, attempt_dir in attempt_directories(mode_dir)
        if (attempt := rebuild_attempt(attempt_dir, iteration, reference_time, scalar_time)) is not None
    ]
    if not attempts:
        return None
    successful = next((attempt for attempt in attempts if attempt.get("ok")), None)
    selected = successful or attempts[-1]
    mode_record = dict(selected)
    mode_record.update(
        {
            "mode": mode,
            "iterations": attempts,
            "max_iterations": max(attempt["iteration"] for attempt in attempts) + 1,
            "ok": successful is not None,
        }
    )
    if successful is not None:
        mode_record["success_iteration"] = successful["iteration"]
    return mode_record


def rebuild_task(task_dir: Path, modes: list[str]) -> dict[str, Any]:
    match = TASK_DIR_PATTERN.fullmatch(task_dir.name)
    if match is None:
        raise ValueError(f"invalid task directory: {task_dir}")
    task_index = int(match.group(1))
    kernel = match.group(2)
    reference = read_json(task_dir / "reference.json")
    reference_parsed = reference.get("parsed") if isinstance(reference.get("parsed"), dict) else {}
    scalar = read_json(task_dir / "scalar_ir.json")
    scalar_parsed = scalar.get("parsed") if isinstance(scalar.get("parsed"), dict) else {}
    reference_time = reference_parsed.get("time_sec")
    scalar_time = scalar_parsed.get("time_sec") if scalar.get("ok") else None
    rebuilt_modes: dict[str, Any] = {}
    for mode in modes:
        mode_dir = task_dir / mode
        if not mode_dir.is_dir():
            continue
        mode_record = rebuild_mode(mode_dir, mode, reference_time, scalar_time)
        if mode_record is not None:
            rebuilt_modes[mode] = mode_record
    record: dict[str, Any] = {
        "task_index": task_index,
        "kernel": kernel,
        "scalar_ir": scalar.get("ir_file"),
        "reference_time_sec": reference_time,
        "scalar_ir_time_sec": scalar_time,
        "expected_checksum": reference_parsed.get("checksum"),
        "modes": rebuilt_modes,
        "ok": any(mode_record.get("ok") for mode_record in rebuilt_modes.values()),
        "rebuilt_from_artifacts": True,
    }
    if not reference.get("ok"):
        record["error"] = "reference build/run failed"
    return record


def update_summary(path: Path, records: list[dict[str, Any]], modes: list[str]) -> None:
    summary = read_json(path)
    summary["tasks"] = len(records)
    summary["modes"] = modes
    summary["records"] = str(path.parent / "records.jsonl")
    summary["successes"] = {
        mode: sum(bool(record.get("modes", {}).get(mode, {}).get("ok")) for record in records)
        for mode in modes
    }
    summary["pass_at_1"] = {
        mode: sum(
            bool(record.get("modes", {}).get(mode, {}).get("iterations"))
            and bool(record["modes"][mode]["iterations"][0].get("ok"))
            for record in records
        )
        for mode in modes
    }
    summary["recovered_by_self_correction"] = {
        mode: sum(
            bool(record.get("modes", {}).get(mode, {}).get("ok"))
            and int(record["modes"][mode].get("success_iteration", 0)) > 0
            for record in records
        )
        for mode in modes
    }
    summary["attempt_zero_validation"] = {}
    for mode in modes:
        first_attempts = [
            record["modes"][mode]["iterations"][0]
            for record in records
            if record.get("modes", {}).get(mode, {}).get("iterations")
        ]
        summary["attempt_zero_validation"][mode] = {
            "seeded": sum(attempt.get("candidate_origin") == "seed" for attempt in first_attempts),
            "tsvc_pass": sum(bool(attempt.get("tsvc_ok")) for attempt in first_attempts),
            "alive2_pass": sum(attempt.get("alive2_status") == "pass" for attempt in first_attempts),
            "differential_fuzz_pass": sum(
                attempt.get("differential_fuzz_status") == "pass" for attempt in first_attempts
            ),
            "accepted": sum(bool(attempt.get("ok")) for attempt in first_attempts),
        }
    for mode in modes:
        times = [
            float(record["modes"][mode]["observed_time_sec"])
            for record in records
            if record.get("modes", {}).get(mode, {}).get("ok")
            and record["modes"][mode].get("observed_time_sec") is not None
        ]
        summary[f"{mode}_mean_time_sec"] = sum(times) / len(times) if times else None
    if "baseline" in modes and "guided" in modes:
        speedups = []
        for record in records:
            baseline = record.get("modes", {}).get("baseline", {})
            guided = record.get("modes", {}).get("guided", {})
            if (
                baseline.get("ok")
                and guided.get("ok")
                and baseline.get("observed_time_sec") is not None
                and guided.get("observed_time_sec") is not None
                and float(guided["observed_time_sec"]) > 0.0
            ):
                speedups.append(float(baseline["observed_time_sec"]) / float(guided["observed_time_sec"]))
        summary["guided_speedup_vs_llm_baseline_mean"] = (
            sum(speedups) / len(speedups) if speedups else None
        )
        summary["guided_speedup_vs_llm_baseline_geomean"] = (
            math.exp(sum(math.log(speedup) for speedup in speedups) / len(speedups)) if speedups else None
        )
    summary["records_rebuilt_from_artifacts"] = True
    atomic_write(path, json.dumps(summary, indent=2, allow_nan=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild TSVC records.jsonl from saved verification artifacts.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Experiment output directory containing task folders.")
    parser.add_argument("--modes", default="baseline,guided", help="Comma-separated mode directory names.")
    parser.add_argument("--no-summary", action="store_true", help="Do not update summary.json.")
    parser.add_argument("--no-backup", action="store_true", help="Do not back up existing index files.")
    args = parser.parse_args()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    task_dirs = []
    for path in args.out_dir.iterdir():
        match = TASK_DIR_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            task_dirs.append((int(match.group(1)), path))
    task_dirs.sort()
    if not task_dirs:
        raise SystemExit(f"no task directories found under {args.out_dir}")

    records = [rebuild_task(path, modes) for _, path in task_dirs]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records_path = args.out_dir / "records.jsonl"
    snapshot_path = args.out_dir / "records.resume_snapshot.jsonl"
    summary_path = args.out_dir / "summary.json"
    if not args.no_backup:
        for path in (records_path, snapshot_path, summary_path):
            backup_existing(path, timestamp)

    content = "".join(json.dumps(record, sort_keys=True, allow_nan=True) + "\n" for record in records)
    atomic_write(records_path, content)
    atomic_write(snapshot_path, content)
    if not args.no_summary:
        update_summary(summary_path, records, modes)

    successes = {
        mode: sum(bool(record.get("modes", {}).get(mode, {}).get("ok")) for record in records)
        for mode in modes
    }
    print(json.dumps({"tasks": len(records), "successes": successes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
