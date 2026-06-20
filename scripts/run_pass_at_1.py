#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = REPO_ROOT / "prompts" / "source_to_asm_fast_v1.txt"


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_path(task: dict[str, str]) -> Path:
    path = Path(task["path"])
    if path.is_absolute():
        return path
    return Path(task["root"]) / path


def render_prompt(
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


def response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]

    chunks: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text", "")
                if text:
                    chunks.append(text)
    return "".join(chunks)


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


def call_openai_responses(
    prompt: str,
    *,
    model: str,
    api_key: str,
    base_url: str,
    reasoning_effort: str,
    max_output_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "input": prompt,
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        base_url.rstrip("/") + "/responses",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenAI API request failed: {exc}") from exc


def evaluate_assembly(
    asm_path: Path,
    target: str,
    workdir: Path,
    cc: str,
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
    ]
    if cc:
        cmd.extend(["--cc", cc])
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pass@1 source/IR-to-assembly experiments with GPT-5.5."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--target", default="x86_64-pc-linux-gnu")
    parser.add_argument("--target-notes", default="System V AMD64 ABI.")
    parser.add_argument(
        "--mode",
        choices=["source-to-asm", "llvm-ir-to-asm"],
        default="source-to-asm",
    )
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "OPENAI_API_KEY is not set; cannot run GPT-5.5 pass@1.",
            file=sys.stderr,
        )
        return 2

    template = args.template.read_text(encoding="utf-8")
    tasks = load_jsonl(args.manifest)
    selected = tasks[args.offset : args.offset + args.limit]
    if not selected:
        parser.error("selected task range is empty")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    run_records: list[dict[str, Any]] = []

    for local_index, task in enumerate(selected):
        global_index = args.offset + local_index
        task_dir = args.out_dir / f"task_{global_index:04d}"
        task_dir.mkdir(parents=True, exist_ok=True)

        src_path = source_path(task)
        source = src_path.read_text(encoding="utf-8", errors="replace")
        prompt = render_prompt(
            template,
            task,
            source,
            args.target,
            args.mode,
            args.target_notes,
        )
        prompt_path = task_dir / "prompt.txt"
        response_path = task_dir / "response.json"
        asm_path = task_dir / "candidate.s"
        result_path = task_dir / "result.json"

        prompt_path.write_text(prompt, encoding="utf-8")
        start = time.time()
        record: dict[str, Any] = {
            "task_index": global_index,
            "task": task,
            "model": args.model,
            "mode": args.mode,
            "target": args.target,
            "template": str(args.template),
            "prompt_path": str(prompt_path),
            "response_path": str(response_path),
            "asm_path": str(asm_path),
            "result_path": str(result_path),
        }

        try:
            response = call_openai_responses(
                prompt,
                model=args.model,
                api_key=api_key,
                base_url=args.api_base_url,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout,
            )
            response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
            assembly = strip_markdown_fence(response_text(response))
            asm_path.write_text(assembly, encoding="utf-8")
            evaluation = evaluate_assembly(asm_path, args.target, task_dir / "eval", args.cc)
            result_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
            record["evaluation"] = evaluation
            record["ok"] = bool(
                evaluation.get("assemble", {}).get("ok")
                and evaluation.get("runner_returncode") == 0
            )
        except Exception as exc:  # noqa: BLE001 - experiment runner records failures.
            record["ok"] = False
            record["error"] = str(exc)

        record["elapsed_sec"] = round(time.time() - start, 3)
        run_records.append(record)
        with records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    total = len(run_records)
    success = sum(1 for row in run_records if row.get("ok"))
    summary = {
        "model": args.model,
        "mode": args.mode,
        "target": args.target,
        "template": str(args.template),
        "tasks": total,
        "pass_at_1_assemble": success / total if total else 0.0,
        "successes": success,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if success == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
