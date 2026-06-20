#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def load_task(manifest: Path, index: int) -> dict[str, str]:
    with manifest.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"manifest has no task at index {index}")


def task_source_path(task: dict[str, str]) -> Path:
    path = Path(task["path"])
    if path.is_absolute():
        return path
    return Path(task["root"]) / path


def build_prompt(task: dict[str, str], source: str, target: str, mode: str) -> str:
    return f"""You are acting as a compiler, not as an explainer.

Task:
- Mode: {mode}
- Source suite: {task["suite"]}
- Task id: {task["task_id"]}
- Input language: {task["language"]}
- Target triple: {target}

Requirements:
- Emit only target assembly.
- Do not include Markdown fences.
- Preserve the observable behavior of the input program or translation unit.
- Follow the platform ABI for the target triple.
- Keep global symbols visible when they are externally referenced.
- Do not call a C, Fortran, or LLVM compiler.

Input:
{source}
"""


def run_agent(agent_cmd: str, prompt: str) -> str:
    proc = subprocess.run(
        shlex.split(agent_cmd),
        input=prompt,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"agent command failed with code {proc.returncode}\n{proc.stderr}"
        )
    return proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask an external LLM compiler agent to generate assembly."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--mode",
        choices=["source-to-asm", "llvm-ir-to-asm"],
        default="source-to-asm",
    )
    parser.add_argument(
        "--agent-cmd",
        required=True,
        help="Command that reads the prompt from stdin and writes assembly to stdout.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--prompt-out",
        type=Path,
        help="Optional path to save the exact prompt sent to the agent.",
    )
    args = parser.parse_args()

    task = load_task(args.manifest, args.index)
    source_path = task_source_path(task)
    source = source_path.read_text(encoding="utf-8", errors="replace")
    prompt = build_prompt(task, source, args.target, args.mode)

    if args.prompt_out:
        args.prompt_out.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_out.write_text(prompt, encoding="utf-8")

    assembly = run_agent(args.agent_cmd, prompt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(assembly, encoding="utf-8")
    print(f"wrote candidate assembly to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
