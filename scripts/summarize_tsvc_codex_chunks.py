#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize parallel TSVC Codex chunk outputs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    chunk_dirs = sorted((args.run_dir / "chunks").glob("chunk_*"))
    records: list[dict] = []
    events: list[dict] = []
    for chunk_dir in chunk_dirs:
        records.extend(load_jsonl(chunk_dir / "records.jsonl"))
        events.extend(load_jsonl(chunk_dir / "events.jsonl"))

    records.sort(key=lambda row: row.get("task_index", -1))
    events.sort(key=lambda row: (row.get("task_index", -1), row.get("iteration", -1)))
    successes = sum(1 for row in records if row.get("ok"))
    first_attempt_successes = sum(
        1
        for row in records
        if row.get("ok") and row.get("success_iteration") == 0
    )
    summary = {
        "tasks": len(records),
        "successes": successes,
        "agentic_success_rate": successes / len(records) if records else 0.0,
        "first_attempt_successes": first_attempt_successes,
        "first_attempt_success_rate": first_attempt_successes / len(records) if records else 0.0,
        "events": len(events),
    }

    (args.run_dir / "records_all.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    (args.run_dir / "events_all.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in events),
        encoding="utf-8",
    )
    (args.run_dir / "summary_all.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# TSVC Codex All-Kernel Summary",
        "",
        f"- Tasks: {summary['tasks']}",
        f"- Successes within budget: {summary['successes']}",
        f"- Agentic success rate: {summary['agentic_success_rate']}",
        f"- First-attempt successes: {summary['first_attempt_successes']}",
        f"- First-attempt success rate: {summary['first_attempt_success_rate']}",
        f"- Attempt events: {summary['events']}",
        "",
        "| Index | Kernel | Success | Iteration | Reference Checksum | Final Checksum | Reference Time (s) | Final Time (s) |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        ref = row.get("reference", {})
        lines.append(
            "| {idx} | {kernel} | {ok} | {iteration} | {refchk} | {finalchk} | {reftime} | {finaltime} |".format(
                idx=row.get("task_index"),
                kernel=row.get("kernel"),
                ok="yes" if row.get("ok") else "no",
                iteration=row.get("success_iteration"),
                refchk=ref.get("checksum"),
                finalchk=row.get("final_checksum"),
                reftime=ref.get("time_sec"),
                finaltime=row.get("final_time_sec"),
            )
        )
    (args.run_dir / "report_all.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
