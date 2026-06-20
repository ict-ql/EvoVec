#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATE_TEMPLATE = REPO_ROOT / "prompts" / "tsvc_kernel_to_asm_fast_v1.txt"
DEFAULT_REPAIR_TEMPLATE = REPO_ROOT / "prompts" / "tsvc_self_correction_fast_v1.txt"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def build_reference(task: dict[str, Any], out_dir: Path, cc: str, timeout: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    support = Path(task["support_dir"])
    exe = out_dir / "reference.exe"
    cmd = [
        cc,
        "-O3",
        "-march=native",
        "-no-pie",
        "-I",
        str(support),
        str(task["reference_kernel"]),
        str(support / "common.c"),
        str(support / "dummy.c"),
        str(support / "globals.c"),
        str(task["harness"]),
        "-lm",
        "-o",
        str(exe),
    ]
    compile_result = run_cmd(cmd, timeout=timeout)
    run_result: dict[str, Any] | None = None
    parsed: dict[str, Any] | None = None
    if compile_result["ok"]:
        run_result = run_cmd([str(exe)], timeout=timeout)
        parsed = parse_tsvc_output(str(run_result["stdout"])) if run_result["ok"] else None
    return {
        "exe": str(exe),
        "compile": compile_result,
        "run": run_result,
        "parsed": parsed,
        "ok": bool(compile_result["ok"] and run_result and run_result["ok"] and parsed and parsed["ok"]),
    }


def build_and_run_candidate(
    task: dict[str, Any],
    asm_path: Path,
    out_dir: Path,
    cc: str,
    expected_checksum: float,
    checksum_atol: float,
    timeout: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    support = Path(task["support_dir"])
    exe = out_dir / "candidate.exe"
    cmd = [
        cc,
        "-O2",
        "-no-pie",
        "-I",
        str(support),
        str(asm_path),
        str(support / "common.c"),
        str(support / "dummy.c"),
        str(support / "globals.c"),
        str(task["harness"]),
        "-lm",
        "-o",
        str(exe),
    ]
    compile_result = run_cmd(cmd, timeout=timeout)
    run_result: dict[str, Any] | None = None
    parsed: dict[str, Any] | None = None
    checksum_ok = False
    if compile_result["ok"]:
        run_result = run_cmd([str(exe)], timeout=timeout)
        parsed = parse_tsvc_output(str(run_result["stdout"])) if run_result["ok"] else None
        if parsed and parsed["ok"]:
            checksum_ok = math.isclose(
                float(parsed["checksum"]),
                expected_checksum,
                rel_tol=0.0,
                abs_tol=checksum_atol,
            )
    return {
        "exe": str(exe),
        "compile": compile_result,
        "run": run_result,
        "parsed": parsed,
        "expected_checksum": expected_checksum,
        "checksum_atol": checksum_atol,
        "checksum_ok": checksum_ok,
        "ok": bool(compile_result["ok"] and run_result and run_result["ok"] and parsed and parsed["ok"] and checksum_ok),
    }


def render_generate_prompt(
    template: str,
    task: dict[str, Any],
    source: str,
    target: str,
    target_notes: str,
) -> str:
    return template.format(
        kernel=task["kernel"],
        target=target,
        target_notes=target_notes,
        iterations=task["iterations"],
        len_1d=task["len_1d"],
        len_2d=task["len_2d"],
        required_symbols=", ".join(task.get("required_symbols", [task["kernel"]])),
        source=source,
    )


def render_repair_prompt(
    template: str,
    task: dict[str, Any],
    source: str,
    candidate: str,
    verifier_result: dict[str, Any],
    target: str,
    target_notes: str,
    iteration: int,
) -> str:
    return template.format(
        kernel=task["kernel"],
        target=target,
        target_notes=target_notes,
        iterations=task["iterations"],
        len_1d=task["len_1d"],
        len_2d=task["len_2d"],
        iteration=iteration,
        required_symbols=", ".join(task.get("required_symbols", [task["kernel"]])),
        source=source,
        candidate=candidate,
        verifier_result=json.dumps(verifier_result, indent=2),
    )


def strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip() + "\n"


def run_agent(agent_cmd: str, prompt: str, timeout: int) -> dict[str, Any]:
    start = time.time()
    try:
        proc = subprocess.run(
            shlex.split(agent_cmd),
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_sec": round(time.time() - start, 3),
            "ok": proc.returncode == 0,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "elapsed_sec": round(time.time() - start, 3),
            "ok": False,
            "error": "agent timeout",
        }


def compact_verifier_feedback(result: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "ok": result.get("ok", False),
        "checksum_ok": result.get("checksum_ok", False),
        "expected_checksum": result.get("expected_checksum"),
        "observed": result.get("parsed"),
    }
    compile_result = result.get("compile") or {}
    run_result = result.get("run") or {}
    compact["compile"] = {
        "ok": compile_result.get("ok"),
        "returncode": compile_result.get("returncode"),
        "stderr": compile_result.get("stderr", "")[-4000:],
    }
    compact["run"] = {
        "ok": run_result.get("ok"),
        "returncode": run_result.get("returncode"),
        "stdout": run_result.get("stdout", "")[-2000:],
        "stderr": run_result.get("stderr", "")[-2000:],
    }
    return compact


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Codex-agent TSVC kernel-to-assembly experiment."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--agent-cmd", default="scripts/codex_asm_agent.sh")
    parser.add_argument("--target", default="x86_64-pc-linux-gnu")
    parser.add_argument("--target-notes", default="System V AMD64 ABI, GNU as syntax, no PIE link.")
    parser.add_argument("--generate-template", type=Path, default=DEFAULT_GENERATE_TEMPLATE)
    parser.add_argument("--repair-template", type=Path, default=DEFAULT_REPAIR_TEMPLATE)
    parser.add_argument("--limit", type=int, default=3, help="Number of tasks to run; <=0 means all remaining tasks.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--build-timeout", type=int, default=60)
    parser.add_argument("--run-timeout", type=int, default=60)
    parser.add_argument("--agent-timeout", type=int, default=900)
    parser.add_argument("--checksum-atol", type=float, default=1e-3)
    parser.add_argument("--reference-only", action="store_true", help="Only build and run compiler reference harnesses; do not invoke the agent.")
    args = parser.parse_args()

    if args.max_iters < 1:
        parser.error("--max-iters must be at least 1")

    tasks = load_jsonl(args.manifest)
    selected = tasks[args.offset :] if args.limit <= 0 else tasks[args.offset : args.offset + args.limit]
    if not selected:
        parser.error("selected task range is empty")

    generate_template = args.generate_template.read_text(encoding="utf-8")
    repair_template = args.repair_template.read_text(encoding="utf-8")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.out_dir / "events.jsonl"
    records_path = args.out_dir / "records.jsonl"
    events_path.write_text("", encoding="utf-8")
    records_path.write_text("", encoding="utf-8")

    records: list[dict[str, Any]] = []
    for task_index, task in enumerate(selected, start=args.offset):
        kernel = task["kernel"]
        task_dir = args.out_dir / f"{task_index:04d}_{kernel}"
        task_dir.mkdir(parents=True, exist_ok=True)
        source = (Path(task["root"]) / task["path"]).read_text(encoding="utf-8", errors="replace")

        reference = build_reference(task, task_dir / "reference", args.cc, args.build_timeout)
        (task_dir / "reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
        if not reference["ok"]:
            record = {
                "task_index": task_index,
                "kernel": kernel,
                "ok": False,
                "error": "reference build/run failed",
                "reference": reference,
            }
            records.append(record)
            with records_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps({"task_index": task_index, "kernel": kernel, "reference_ok": False, "ok": False}), flush=True)
            continue

        expected = float(reference["parsed"]["checksum"])
        if args.reference_only:
            record = {
                "task_index": task_index,
                "kernel": kernel,
                "ok": True,
                "reference_only": True,
                "reference": {
                    "checksum": expected,
                    "time_sec": reference["parsed"]["time_sec"],
                    "json": str(task_dir / "reference.json"),
                },
                "task": task,
            }
            records.append(record)
            with records_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps({"task_index": task_index, "kernel": kernel, "reference_ok": True, "checksum": expected, "time_sec": reference["parsed"]["time_sec"]}), flush=True)
            continue

        candidate = ""
        verifier_result: dict[str, Any] | None = None
        record: dict[str, Any] = {
            "task_index": task_index,
            "kernel": kernel,
            "task": task,
            "reference": {
                "checksum": expected,
                "time_sec": reference["parsed"]["time_sec"],
                "json": str(task_dir / "reference.json"),
            },
            "iterations": [],
            "ok": False,
        }

        for iteration in range(args.max_iters):
            iter_dir = task_dir / f"iter_{iteration:02d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            if iteration == 0:
                prompt = render_generate_prompt(
                    generate_template, task, source, args.target, args.target_notes
                )
            else:
                assert verifier_result is not None
                prompt = render_repair_prompt(
                    repair_template,
                    task,
                    source,
                    candidate,
                    compact_verifier_feedback(verifier_result),
                    args.target,
                    args.target_notes,
                    iteration,
                )

            prompt_path = iter_dir / "prompt.txt"
            agent_path = iter_dir / "agent.json"
            asm_path = iter_dir / "candidate.s"
            verify_path = iter_dir / "verify.json"
            prompt_path.write_text(prompt, encoding="utf-8")

            agent_result = run_agent(args.agent_cmd, prompt, args.agent_timeout)
            agent_path.write_text(json.dumps(agent_result, indent=2), encoding="utf-8")
            event: dict[str, Any] = {
                "task_index": task_index,
                "kernel": kernel,
                "iteration": iteration,
                "prompt_path": str(prompt_path),
                "agent_path": str(agent_path),
                "asm_path": str(asm_path),
                "verify_path": str(verify_path),
                "agent_ok": agent_result["ok"],
                "agent_elapsed_sec": agent_result["elapsed_sec"],
            }

            if not agent_result["ok"]:
                event["ok"] = False
                event["error"] = agent_result.get("error") or agent_result.get("stderr")
                record["iterations"].append(event)
                with events_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, sort_keys=True) + "\n")
                print(json.dumps(event), flush=True)
                break

            candidate = strip_markdown_fence(str(agent_result["stdout"]))
            asm_path.write_text(candidate, encoding="utf-8")
            verifier_result = build_and_run_candidate(
                task,
                asm_path,
                iter_dir / "candidate",
                args.cc,
                expected,
                args.checksum_atol,
                args.run_timeout,
            )
            verify_path.write_text(json.dumps(verifier_result, indent=2), encoding="utf-8")

            parsed = verifier_result.get("parsed") or {}
            event.update(
                {
                    "ok": verifier_result["ok"],
                    "compile_ok": verifier_result["compile"]["ok"],
                    "run_ok": bool(verifier_result.get("run") and verifier_result["run"]["ok"]),
                    "checksum_ok": verifier_result["checksum_ok"],
                    "expected_checksum": expected,
                    "observed_checksum": parsed.get("checksum"),
                    "observed_time_sec": parsed.get("time_sec"),
                    "reference_time_sec": reference["parsed"]["time_sec"],
                }
            )
            record["iterations"].append(event)
            with events_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event), flush=True)

            if verifier_result["ok"]:
                record["ok"] = True
                record["success_iteration"] = iteration
                record["final_checksum"] = parsed.get("checksum")
                record["final_time_sec"] = parsed.get("time_sec")
                break

        records.append(record)
        with records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    successes = sum(1 for item in records if item.get("ok"))
    summary = {
        "tasks": len(records),
        "successes": successes,
        "agentic_success_rate": successes / len(records) if records else 0.0,
        "max_iters": args.max_iters,
        "reference_only": args.reference_only,
        "events": str(events_path),
        "records": str(records_path),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if successes == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
