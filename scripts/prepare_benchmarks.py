#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neural_asm_pipeline.tasks import CompileTask, detect_language


def iter_tasks(suite: str, root: Path) -> list[CompileTask]:
    tasks: list[CompileTask] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        language = detect_language(path)
        if language is None:
            continue
        if path.name.startswith("."):
            continue
        rel = path.relative_to(root)
        task_id = f"{suite}:{rel.as_posix()}"
        tasks.append(
            CompileTask(
                suite=suite,
                task_id=task_id,
                path=path,
                language=language,
                granularity="translation_unit",
            )
        )
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a JSONL manifest of NPB/TSVC source files."
    )
    parser.add_argument("--suite", choices=["npb", "tsvc"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        parser.error(f"benchmark root does not exist: {root}")
    if not root.is_dir():
        parser.error(f"benchmark root is not a directory: {root}")

    tasks = iter_tasks(args.suite, root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for task in tasks:
            record = task.to_json(root)
            record["root"] = str(root)
            f.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"wrote {len(tasks)} tasks to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
