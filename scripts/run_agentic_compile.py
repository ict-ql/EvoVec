#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATE_TEMPLATE = REPO_ROOT / "prompts" / "source_to_asm_fast_v1.txt"
DEFAULT_REPAIR_TEMPLATE = REPO_ROOT / "prompts" / "self_correction_fast_v1.txt"


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def task_source_path(task: dict[str, str]) -> Path:
    path = Path(task["path"])
    if path.is_absolute():
        return path
    return Path(task["root"]) / path


def render_generate_prompt(
    template: str,
    task: dict[str, str],
    source: str,
    target: str,
    mode: str,
    target_notes: str,
) -> str:
    return template.format(
        mode=mode,
        suite=task["suite"],
        task_id=task["task_id"],
        language=task["language"],
        target=target,
        target_notes=target_notes,
        source=source,
    )


def render_repair_prompt(
    template: str,
    task: dict[str, str],
    source: str,
    candidate: str,
    verifier_result: dict[str, Any],
    target: str,
    mode: str,
    target_notes: str,
    iteration: int,
) -> str:
    return template.format(
        mode=mode,
        suite=task["suite"],
        task_id=task["task_id"],
        language=task["language"],
        target=target,
        target_notes=target_notes,
        iteration=iteration,
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


def evaluate_assembly(
    asm_path: Path,
    target: str,
    workdir: Path,
    cc: str,
    link_args: list[str] | None,
    run_command: str | None,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_asm.py"),
        "--asm",
        str(asm_path),
        "--target",
        target,
        "--workdir",
        str(workdir),
        "--cc",
        cc,
    ]
    if link_args is not None:
        cmd.append("--link")
        cmd.extend(link_args)
    if run_command:
        cmd.extend(["--run-command", run_command])

    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError:
        parsed = {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    parsed["runner_returncode"] = proc.returncode
    return parsed


def verifier_passed(result: dict[str, Any]) -> bool:
    return bool(
        result.get("runner_returncode") == 0
        and result.get("assemble", {}).get("ok")
        and result.get("link", {"ok": True}).get("ok")
        and result.get("run", {"ok": True}).get("ok")
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an iterative LLM compiler agent with fixed repair prompts."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--agent-cmd", required=True)
    parser.add_argument("--target", default="x86_64-pc-linux-gnu")
    parser.add_argument("--target-notes", default="System V AMD64 ABI.")
    parser.add_argument(
        "--mode",
        choices=["source-to-asm", "llvm-ir-to-asm"],
        default="source-to-asm",
    )
    parser.add_argument("--generate-template", type=Path, default=DEFAULT_GENERATE_TEMPLATE)
    parser.add_argument("--repair-template", type=Path, default=DEFAULT_REPAIR_TEMPLATE)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--agent-timeout", type=int, default=600)
    parser.add_argument("--cc", default="cc")
    parser.add_argument(
        "--link-arg",
        action="append",
        default=None,
        help="One linker file/flag. Repeat for multiple args. Enables link verification.",
    )
    parser.add_argument(
        "--run-command",
        help="Command to execute after linking. Use {exe} for the generated executable.",
    )
    args = parser.parse_args()

    if args.max_iters < 1:
        parser.error("--max-iters must be at least 1")

    tasks = load_jsonl(args.manifest)
    selected = tasks[args.offset : args.offset + args.limit]
    if not selected:
        parser.error("selected task range is empty")

    generate_template = args.generate_template.read_text(encoding="utf-8")
    repair_template = args.repair_template.read_text(encoding="utf-8")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    records: list[dict[str, Any]] = []

    for local_index, task in enumerate(selected):
        global_index = args.offset + local_index
        task_dir = args.out_dir / f"task_{global_index:04d}"
        task_dir.mkdir(parents=True, exist_ok=True)
        source = task_source_path(task).read_text(encoding="utf-8", errors="replace")

        candidate = ""
        verifier_result: dict[str, Any] | None = None
        task_record: dict[str, Any] = {
            "task_index": global_index,
            "task": task,
            "mode": args.mode,
            "target": args.target,
            "generate_template": str(args.generate_template),
            "repair_template": str(args.repair_template),
            "max_iters": args.max_iters,
            "iterations": [],
            "ok": False,
        }

        for iteration in range(args.max_iters):
            iter_dir = task_dir / f"iter_{iteration:02d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            if iteration == 0:
                prompt = render_generate_prompt(
                    generate_template,
                    task,
                    source,
                    args.target,
                    args.mode,
                    args.target_notes,
                )
            else:
                assert verifier_result is not None
                prompt = render_repair_prompt(
                    repair_template,
                    task,
                    source,
                    candidate,
                    verifier_result,
                    args.target,
                    args.mode,
                    args.target_notes,
                    iteration,
                )

            prompt_path = iter_dir / "prompt.txt"
            asm_path = iter_dir / "candidate.s"
            agent_path = iter_dir / "agent.json"
            result_path = iter_dir / "result.json"
            prompt_path.write_text(prompt, encoding="utf-8")

            iteration_record: dict[str, Any] = {
                "iteration": iteration,
                "prompt_path": str(prompt_path),
                "asm_path": str(asm_path),
                "agent_path": str(agent_path),
                "result_path": str(result_path),
            }

            try:
                agent_result = run_agent(args.agent_cmd, prompt, args.agent_timeout)
            except subprocess.TimeoutExpired as exc:
                agent_result = {
                    "returncode": 124,
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                    "elapsed_sec": args.agent_timeout,
                    "ok": False,
                    "error": "agent timeout",
                }

            agent_path.write_text(json.dumps(agent_result, indent=2), encoding="utf-8")
            if not agent_result["ok"]:
                iteration_record["agent_ok"] = False
                iteration_record["error"] = agent_result.get("error") or agent_result["stderr"]
                task_record["iterations"].append(iteration_record)
                break

            candidate = strip_markdown_fence(str(agent_result["stdout"]))
            asm_path.write_text(candidate, encoding="utf-8")

            verifier_result = evaluate_assembly(
                asm_path,
                args.target,
                iter_dir / "eval",
                args.cc,
                args.link_arg,
                args.run_command,
            )
            result_path.write_text(json.dumps(verifier_result, indent=2), encoding="utf-8")

            iteration_record["agent_ok"] = True
            iteration_record["verifier_passed"] = verifier_passed(verifier_result)
            iteration_record["assemble_ok"] = bool(
                verifier_result.get("assemble", {}).get("ok")
            )
            iteration_record["link_ok"] = bool(
                verifier_result.get("link", {"ok": True}).get("ok")
            )
            iteration_record["run_ok"] = bool(
                verifier_result.get("run", {"ok": True}).get("ok")
            )
            task_record["iterations"].append(iteration_record)

            if iteration_record["verifier_passed"]:
                task_record["ok"] = True
                task_record["success_iteration"] = iteration
                break

        records.append(task_record)
        with records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(task_record, sort_keys=True) + "\n")

    total = len(records)
    success = sum(1 for row in records if row.get("ok"))
    summary = {
        "mode": args.mode,
        "target": args.target,
        "tasks": total,
        "max_iters": args.max_iters,
        "successes": success,
        "agentic_success_rate": success / total if total else 0.0,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if success == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
