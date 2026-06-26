#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def select_tasks(tasks: list[dict[str, Any]], kernels: str) -> list[dict[str, Any]]:
    if kernels.strip().lower() == "all":
        return tasks
    wanted = [item.strip() for item in kernels.split(",") if item.strip()]
    wanted_set = set(wanted)
    selected = [task for task in tasks if str(task.get("kernel")) in wanted_set]
    missing = sorted(wanted_set.difference(str(task.get("kernel")) for task in selected))
    if missing:
        raise SystemExit(f"kernels not found in manifest: {', '.join(missing)}")
    return selected


def resolve_executable(executable: str, *, dry_run: bool = False) -> str:
    path = Path(executable)
    if path.exists():
        return str(path)
    resolved = shutil.which(executable)
    if resolved:
        return resolved
    if dry_run:
        return executable
    raise SystemExit(
        f"could not find executable {executable!r}; pass an explicit path with the matching flag"
    )


def display_command(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_checked(command: list[str], *, dry_run: bool) -> None:
    print(display_command(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def emit_ir_for_tasks(args: argparse.Namespace, tasks: list[dict[str, Any]]) -> list[Path]:
    clang = resolve_executable(args.clang, dry_run=args.dry_run)
    args.ir_dir.mkdir(parents=True, exist_ok=True)
    emitted: list[Path] = []
    for task in tasks:
        kernel = str(task.get("kernel", "")).strip()
        if not kernel:
            raise SystemExit(f"manifest row is missing kernel: {task}")
        tsvc_root = resolve_tsvc_root(args, task)
        reference_kernel = resolve_task_path(
            task,
            "reference_kernel",
            tsvc_root / "kernels" / kernel / "reference_kernel.c",
        )
        support_dir = resolve_task_path(task, "support_dir", tsvc_root / "support")
        if not reference_kernel.exists():
            raise SystemExit(f"missing reference kernel for {kernel}: {reference_kernel}")
        if not support_dir.exists():
            raise SystemExit(f"missing support directory for {kernel}: {support_dir}")
        out_ir = args.ir_dir / f"{kernel}.ll"
        emitted.append(out_ir)
        if args.skip_existing_ir and out_ir.exists():
            continue
        command = [
            clang,
            "-S",
            "-emit-llvm",
            f"-O{args.opt_level}",
            "-fno-vectorize",
            "-fno-slp-vectorize",
            "-fno-unroll-loops",
            "-I",
            str(support_dir),
        ]
        if args.target:
            command.extend(["--target", args.target])
        command.extend(args.extra_clang_arg)
        command.extend([str(reference_kernel), "-o", str(out_ir)])
        run_checked(command, dry_run=args.dry_run)
    return emitted


def resolve_tsvc_root(args: argparse.Namespace, task: dict[str, Any]) -> Path:
    if args.tsvc_root:
        return args.tsvc_root
    task_root = Path(str(task.get("root", "")))
    if task_root.exists():
        return task_root
    return REPO_ROOT / "runs/tsvc_kernels/v1"


def resolve_task_path(task: dict[str, Any], key: str, fallback: Path) -> Path:
    raw = str(task.get(key, "")).strip()
    if raw:
        path = Path(raw)
        if path.exists():
            return path
    return fallback


def run_recommender(args: argparse.Namespace) -> None:
    python = resolve_executable(args.python, dry_run=args.dry_run)
    recommender = args.recommender_script
    if not recommender.exists():
        raise SystemExit(f"missing recommender script: {recommender}")
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --emit-ir-only is used")
    if not args.checkpoint.exists():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    if not args.instruction_graph_dir:
        raise SystemExit("--instruction-graph-dir is required unless --emit-ir-only is used")
    if not args.instruction_graph_dir.exists():
        raise SystemExit(f"missing instruction graph directory: {args.instruction_graph_dir}")
    if not args.dry_run and not any(args.ir_dir.glob(args.glob)):
        raise SystemExit(f"no IR files matching {args.glob!r} in {args.ir_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.out_dir / args.out_jsonl_name
    out_md = args.out_dir / args.out_md_name
    command = [
        python,
        str(recommender),
        "--ir-dir",
        str(args.ir_dir),
        "--glob",
        args.glob,
        "--checkpoint",
        str(args.checkpoint),
        "--instruction-graph-dir",
        str(args.instruction_graph_dir),
        "--out-jsonl",
        str(out_jsonl),
        "--out-md",
        str(out_md),
        "--top-k",
        str(args.top_k),
        "--device",
        args.device,
        "--candidate-modes",
        args.candidate_modes,
        "--store-slice-depth",
        str(args.store_slice_depth),
        "--store-slice-max-nodes",
        str(args.store_slice_max_nodes),
        "--max-report-regions",
        str(args.max_report_regions),
    ]
    if args.max_regions_per_function > 0:
        command.extend(["--max-regions-per-function", str(args.max_regions_per_function)])
    if args.include_memory_labels:
        command.append("--include-memory-labels")
    if args.only_predicted_change:
        command.extend(["--only-predicted-change", "--change-threshold", str(args.change_threshold)])
    run_checked(command, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit pre-vectorization TSVC IR and run top-k region intrinsic inference."
    )
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "runs/manifests/tsvc_kernels.jsonl")
    parser.add_argument("--kernels", default="all", help="Comma-separated kernel names, or 'all'.")
    parser.add_argument(
        "--tsvc-root",
        type=Path,
        help="Root produced by prepare_tsvc_kernels.py; useful when the manifest was copied from another machine.",
    )
    parser.add_argument("--ir-dir", type=Path, default=REPO_ROOT / "runs/tsvc_prevec_ir")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs/tsvc_region_recommendations/top20")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--instruction-graph-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--target")
    parser.add_argument("--opt-level", default="3", choices=["0", "1", "2", "3", "s", "z"])
    parser.add_argument("--extra-clang-arg", action="append", default=[])
    parser.add_argument("--skip-existing-ir", action="store_true")
    parser.add_argument("--skip-emit-ir", action="store_true")
    parser.add_argument("--emit-ir-only", action="store_true")
    parser.add_argument("--glob", default="*.ll")
    parser.add_argument(
        "--recommender-script",
        type=Path,
        default=REPO_ROOT / "scripts/recommend_region_intrinsics.py",
    )
    parser.add_argument("--candidate-modes", default="store-backward")
    parser.add_argument("--store-slice-depth", type=int, default=5)
    parser.add_argument("--store-slice-max-nodes", type=int, default=128)
    parser.add_argument("--max-regions-per-function", type=int, default=0)
    parser.add_argument("--max-report-regions", type=int, default=50)
    parser.add_argument("--include-memory-labels", action="store_true")
    parser.add_argument("--only-predicted-change", action="store_true")
    parser.add_argument("--change-threshold", type=float, default=0.0)
    parser.add_argument("--out-jsonl-name", default="recommendations.jsonl")
    parser.add_argument("--out-md-name", default="recommendations.md")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if not args.manifest.exists():
        raise SystemExit(
            f"missing TSVC manifest: {args.manifest}; run scripts/prepare_tsvc_kernels.py first"
        )

    tasks = select_tasks(load_jsonl(args.manifest), args.kernels)
    if not tasks:
        raise SystemExit(f"no TSVC tasks selected from {args.manifest}")
    print(f"selected {len(tasks)} TSVC kernels", flush=True)

    if not args.skip_emit_ir:
        emitted = emit_ir_for_tasks(args, tasks)
        print(f"IR directory: {args.ir_dir} ({len(emitted)} files requested)", flush=True)
    if not args.emit_ir_only:
        run_recommender(args)
        print(f"recommendations: {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
