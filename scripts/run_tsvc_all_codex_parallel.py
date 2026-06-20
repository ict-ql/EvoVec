#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]


def manifest_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def chunk_ranges(total: int, chunks: int) -> list[tuple[int, int]]:
    base = total // chunks
    rem = total % chunks
    ranges: list[tuple[int, int]] = []
    start = 0
    for i in range(chunks):
        size = base + (1 if i < rem else 0)
        if size:
            ranges.append((start, size))
            start += size
    return ranges


def read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def count_events(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def write_status(out_dir: Path, processes: list[dict], started_at: float) -> dict:
    chunks = []
    total_events = 0
    done = 0
    for proc_info in processes:
        chunk_dir = Path(proc_info["out_dir"])
        summary = read_summary(chunk_dir / "summary.json")
        events = count_events(chunk_dir / "events.jsonl")
        total_events += events
        returncode = proc_info["process"].poll()
        is_done = returncode is not None
        done += 1 if is_done else 0
        chunks.append(
            {
                "chunk": proc_info["chunk"],
                "offset": proc_info["offset"],
                "limit": proc_info["limit"],
                "pid": proc_info["process"].pid,
                "returncode": returncode,
                "done": is_done,
                "events": events,
                "summary": summary,
                "out_dir": str(chunk_dir),
                "log": proc_info["log"],
            }
        )
    status = {
        "elapsed_sec": round(time.time() - started_at, 1),
        "chunks_total": len(processes),
        "chunks_done": done,
        "events_total": total_events,
        "chunks": chunks,
    }
    (out_dir / "parallel_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all TSVC kernels in parallel chunks.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--agent-cmd", default="scripts/codex_asm_agent.sh")
    parser.add_argument("--run-timeout", type=int, default=30)
    parser.add_argument("--build-timeout", type=int, default=30)
    parser.add_argument("--agent-timeout", type=int, default=900)
    parser.add_argument("--poll-sec", type=int, default=30)
    args = parser.parse_args()

    total = manifest_count(args.manifest)
    ranges = chunk_ranges(total, args.chunks)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    processes: list[dict] = []
    for chunk_id, (offset, limit) in enumerate(ranges):
        chunk_dir = chunks_dir / f"chunk_{chunk_id:02d}"
        log_path = logs_dir / f"chunk_{chunk_id:02d}.log"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_tsvc_codex_experiment.py"),
            "--manifest",
            str(args.manifest),
            "--out-dir",
            str(chunk_dir),
            "--agent-cmd",
            args.agent_cmd,
            "--offset",
            str(offset),
            "--limit",
            str(limit),
            "--max-iters",
            str(args.max_iters),
            "--run-timeout",
            str(args.run_timeout),
            "--build-timeout",
            str(args.build_timeout),
            "--agent-timeout",
            str(args.agent_timeout),
        ]
        log_file = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append(
            {
                "chunk": chunk_id,
                "offset": offset,
                "limit": limit,
                "process": proc,
                "out_dir": str(chunk_dir),
                "log": str(log_path),
                "log_file": log_file,
            }
        )

    started_at = time.time()
    try:
        while True:
            status = write_status(args.out_dir, processes, started_at)
            print(json.dumps({k: status[k] for k in ("elapsed_sec", "chunks_done", "chunks_total", "events_total")}), flush=True)
            if status["chunks_done"] == status["chunks_total"]:
                break
            time.sleep(args.poll_sec)
    finally:
        for proc_info in processes:
            proc_info["log_file"].close()

    failures = [p for p in processes if p["process"].returncode not in (0, None)]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
