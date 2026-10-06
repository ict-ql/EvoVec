#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_tsvc_output(stdout: str) -> dict[str, float]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty TSVC output")
    columns = lines[-1].split()
    if len(columns) < 2:
        raise ValueError(f"could not parse TSVC output: {lines[-1]!r}")
    time_sec = float(columns[-2])
    checksum = float(columns[-1])
    if not math.isfinite(time_sec) or time_sec < 0.0:
        raise ValueError(f"invalid TSVC time: {columns[-2]!r}")
    return {"time_sec": time_sec, "checksum": checksum}


def checksum_equal(left: float, right: float, atol: float) -> bool:
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    if math.isinf(left) or math.isinf(right):
        return left == right
    return math.isclose(left, right, rel_tol=0.0, abs_tol=atol)


def accepted(mode_record: dict[str, Any]) -> bool:
    semantic_pass = (
        mode_record.get("alive2_status") == "pass"
        or mode_record.get("differential_fuzz_status") == "pass"
    )
    return bool(mode_record.get("ok") and mode_record.get("tsvc_ok") and semantic_pass)


def candidate_path(out_dir: Path, record: dict[str, Any], mode: str) -> Path:
    mode_record = record["modes"][mode]
    stored = Path(str(mode_record.get("candidate_ir", "")))
    if stored.exists():
        return stored
    task_dir = out_dir / f"{int(record['task_index']):04d}_{record['kernel']}" / mode
    iteration = int(mode_record.get("success_iteration", mode_record.get("iteration", 0)))
    return (task_dir if iteration == 0 else task_dir / f"iter_{iteration:02d}") / "candidate.ll"


def run_command(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)


def compile_program(command: list[str], timeout: int) -> None:
    result = run_command(command, timeout)
    if result.returncode != 0:
        raise RuntimeError(f"compile failed ({result.returncode}): {' '.join(command)}\n{result.stderr}")


def run_program(executable: Path, timeout: int, taskset_core: int | None) -> dict[str, float]:
    command = [str(executable)]
    if taskset_core is not None:
        command = ["taskset", "-c", str(taskset_core), *command]
    result = run_command(command, timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"run failed ({result.returncode}): {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return parse_tsvc_output(result.stdout)


def compile_commands(
    args: argparse.Namespace,
    record: dict[str, Any],
    build_dir: Path,
    accepted_modes: list[str],
) -> dict[str, list[str]]:
    kernel = str(record["kernel"])
    kernel_dir = args.tsvc_root / "kernels" / kernel
    support = args.tsvc_root / "support"
    common_sources = [
        str(support / "common.c"),
        str(support / "dummy.c"),
        str(support / "globals.c"),
        str(kernel_dir / "harness.c"),
    ]
    commands = {
        "reference": [
            args.gcc,
            "-O3",
            f"-march={args.march}",
            "-no-pie",
            "-I",
            str(support),
            str(kernel_dir / "reference_kernel.c"),
            *common_sources,
            "-lm",
            "-o",
            str(build_dir / "reference.exe"),
        ]
    }
    for mode in accepted_modes:
        commands[mode] = [
            args.clang,
            "-O3",
            f"-march={args.march}",
            "-no-pie",
            "-I",
            str(support),
            str(candidate_path(args.experiment_dir, record, mode)),
            *common_sources,
            "-lm",
            "-o",
            str(build_dir / f"{mode}.exe"),
        ]
    return commands


def benchmark_kernel(
    args: argparse.Namespace,
    record: dict[str, Any],
    accepted_modes: list[str],
) -> dict[str, Any]:
    kernel = str(record["kernel"])
    task_index = int(record["task_index"])
    build_dir = args.output_dir / "build" / f"{task_index:04d}_{kernel}"
    build_dir.mkdir(parents=True, exist_ok=True)
    commands = compile_commands(args, record, build_dir, accepted_modes)
    for command in commands.values():
        compile_program(command, args.build_timeout)

    names = ["reference", *accepted_modes]
    executables = {name: build_dir / f"{name}.exe" for name in names}
    for _ in range(args.warmups):
        for name in names:
            run_program(executables[name], args.run_timeout, args.taskset_core)

    samples: dict[str, list[dict[str, float]]] = {name: [] for name in executables}
    for repetition in range(args.repetitions):
        offset = repetition % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            samples[name].append(run_program(executables[name], args.run_timeout, args.taskset_core))

    reference_checksum = samples["reference"][0]["checksum"]
    checksum_ok = {
        name: all(checksum_equal(sample["checksum"], reference_checksum, args.checksum_atol) for sample in values)
        for name, values in samples.items()
    }
    medians = {
        name: statistics.median(sample["time_sec"] for sample in values) for name, values in samples.items()
    }
    speedup = {
        f"{mode}_vs_reference": medians["reference"] / medians[mode] for mode in accepted_modes
    }
    if "baseline" in accepted_modes and "guided" in accepted_modes:
        speedup["guided_vs_baseline"] = medians["baseline"] / medians["guided"]
    return {
        "task_index": task_index,
        "kernel": kernel,
        "ok": all(checksum_ok.values()),
        "accepted_modes": accepted_modes,
        "compiler": {"reference": f"{args.gcc} -O3", "baseline": args.clang, "guided": args.clang},
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "samples": samples,
        "median_time_sec": medians,
        "checksum_ok": checksum_ok,
        "speedup": speedup,
        "compile_commands": commands,
    }


def write_aggregate(args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    results.sort(key=lambda result: int(result["task_index"]))
    jsonl = "".join(json.dumps(result, sort_keys=True, allow_nan=True) + "\n" for result in results)
    (args.output_dir / "performance_records.jsonl").write_text(jsonl, encoding="utf-8")
    with (args.output_dir / "performance.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "task_index",
                "kernel",
                "reference_median_sec",
                "baseline_median_sec",
                "guided_median_sec",
                "baseline_vs_reference",
                "guided_vs_reference",
                "guided_vs_baseline",
            ]
        )
        for result in results:
            medians = result["median_time_sec"]
            speedup = result["speedup"]
            writer.writerow(
                [
                    result["task_index"],
                    result["kernel"],
                    medians["reference"],
                    medians.get("baseline"),
                    medians.get("guided"),
                    speedup.get("baseline_vs_reference"),
                    speedup.get("guided_vs_reference"),
                    speedup.get("guided_vs_baseline"),
                ]
            )
    valid = [result for result in results if result.get("ok")]
    summary: dict[str, Any] = {
        "kernels": len(results),
        "valid_kernels": len(valid),
        "selection": "union of independently accepted baseline and guided candidates",
        "accepted_baseline": sum("baseline" in result.get("accepted_modes", []) for result in results),
        "accepted_guided": sum("guided" in result.get("accepted_modes", []) for result in results),
        "paired_baseline_guided": sum(
            "baseline" in result.get("accepted_modes", []) and "guided" in result.get("accepted_modes", [])
            for result in results
        ),
        "reference": f"{args.gcc} -O3 -march={args.march}",
        "candidate_compiler": f"{args.clang} -O3 -march={args.march}",
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "taskset_core": args.taskset_core,
    }
    for metric in ("baseline_vs_reference", "guided_vs_reference", "guided_vs_baseline"):
        values = [
            float(result["speedup"][metric])
            for result in valid
            if result.get("speedup", {}).get(metric) is not None
        ]
        summary[f"{metric}_geomean"] = (
            math.exp(sum(math.log(value) for value in values) / len(values)) if values else None
        )
        summary[f"{metric}_median"] = statistics.median(values) if values else None
    (args.output_dir / "performance_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark accepted TSVC baseline/guided IR against GCC -O3.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--tsvc-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gcc", default="gcc")
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--march", default="native")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--taskset-core", type=int)
    parser.add_argument("--checksum-atol", type=float, default=1e-3)
    parser.add_argument("--build-timeout", type=int, default=120)
    parser.add_argument("--run-timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.warmups < 0 or args.repetitions <= 0:
        raise SystemExit("warmups must be non-negative and repetitions must be positive")
    records = load_jsonl(args.experiment_dir / "records.jsonl")
    selected = []
    for record in records:
        accepted_modes = [
            mode for mode in ("baseline", "guided") if accepted(record.get("modes", {}).get(mode, {}))
        ]
        if accepted_modes:
            selected.append((record, accepted_modes))
    print(
        json.dumps(
            {
                "records": len(records),
                "accepted_baseline": sum(
                    accepted(record.get("modes", {}).get("baseline", {})) for record in records
                ),
                "accepted_guided": sum(
                    accepted(record.get("modes", {}).get("guided", {})) for record in records
                ),
                "benchmark_union_kernels": len(selected),
                "paired_baseline_guided_kernels": sum(len(modes) == 2 for _, modes in selected),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = args.output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for position, (record, accepted_modes) in enumerate(selected, start=1):
        result_path = result_dir / f"{int(record['task_index']):04d}_{record['kernel']}.json"
        if args.resume and result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            print(f"[{position}/{len(selected)}] benchmarking {record['kernel']}", flush=True)
            result = benchmark_kernel(args, record, accepted_modes)
            result_path.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        results.append(result)
        write_aggregate(args, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
