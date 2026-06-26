#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


CONFIG_ARG_MAP = {
    "batch_size": "--batch-size",
    "instruction_batch_size": "--instruction-batch-size",
    "hidden_dim": "--hidden-dim",
    "layers": "--layers",
    "max_edge_position": "--max-edge-position",
    "max_edge_count": "--max-edge-count",
    "dropout": "--dropout",
    "lr": "--lr",
    "weight_decay": "--weight-decay",
    "temperature": "--temperature",
    "match_loss_weight": "--match-loss-weight",
    "max_pos_weight": "--max-pos-weight",
    "selection_split": "--selection-split",
    "seed": "--seed",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def training_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.train_config is not None:
        return read_json(args.train_config)
    return read_json(args.model_dir / "config.json")


def instruction_labels(graph_dir: Path) -> list[str]:
    labels: list[str] = []
    for path in sorted(graph_dir.glob("*.instruction_graphs.jsonl")):
        for row in load_jsonl(path):
            text_parts = [
                str(row.get("label", "")),
                str(row.get("instruction", "")),
                str(row.get("intrinsic", "")),
                str(row.get("extension", "")),
            ]
            labels.append(" ".join(part for part in text_parts if part))
    return labels


def count_matching_labels(graph_dir: Path, pattern: re.Pattern[str]) -> int:
    return sum(1 for label in instruction_labels(graph_dir) if pattern.search(label))


def run_command(cmd: list[str], cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def append_config_args(cmd: list[str], config: dict[str, Any]) -> None:
    for key, flag in CONFIG_ARG_MAP.items():
        if key in config and config[key] is not None:
            cmd.extend([flag, str(config[key])])
    if "eval_ks" in config and config["eval_ks"] is not None:
        value = config["eval_ks"]
        if isinstance(value, list):
            value = ",".join(str(item) for item in value)
        cmd.extend(["--eval-ks", str(value)])


def compile_gemm_ir(args: argparse.Namespace, root: Path) -> Path:
    source = args.gemm_source
    ir = args.gemm_ir
    if not source.is_absolute():
        source = root / source
    if not ir.is_absolute():
        ir = root / ir
    if args.skip_gemm_compile:
        if not ir.exists():
            raise SystemExit(f"--skip-gemm-compile was passed but IR does not exist: {ir}")
        return ir
    if ir.exists() and not args.force_gemm_compile:
        return ir
    ir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.clang,
        "-S",
        "-emit-llvm",
        args.gemm_opt_level,
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-fno-unroll-loops",
        str(source),
        "-o",
        str(ir),
    ]
    for extra in args.clang_extra_arg:
        cmd.insert(-2, extra)
    run_command(cmd, root)
    return ir


def latest_epoch(model_dir: Path) -> int | None:
    history_path = model_dir / "history.json"
    if not history_path.exists():
        return None
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(history, list) or not history:
        return None
    return int(history[-1].get("epoch", 0))


def train_one_round(args: argparse.Namespace, root: Path, round_index: int) -> None:
    checkpoint = args.model_dir / "last.pt"
    config = training_config(args)
    cmd = [
        sys.executable,
        "scripts/train_region_intrinsic_model.py",
        "--region-dir",
        str(args.region_dir),
        "--instruction-graph-dir",
        str(args.train_instruction_graph_dir),
        "--out-dir",
        str(args.model_dir),
        "--device",
        args.device,
    ]
    if checkpoint.exists():
        cmd.extend(
            [
                "--resume-from",
                str(checkpoint),
                "--resume-extra-epochs",
                str(args.epochs_per_round),
            ]
        )
        mode = f"+{args.epochs_per_round} epochs"
    else:
        cmd.extend(["--epochs", str(args.epochs_per_round)])
        mode = f"fresh {args.epochs_per_round} epochs"
    append_config_args(cmd, config)
    print(f"\n=== train round {round_index}: {mode} ===", flush=True)
    run_command(cmd, root)


def recommendation_matches_target(item: dict[str, Any], pattern: re.Pattern[str]) -> bool:
    text = " ".join(
        str(item.get(key, ""))
        for key in ("label", "instruction", "intrinsic", "extension", "architecture")
    )
    return bool(pattern.search(text))


def summarize_recommendations(path: Path, pattern: re.Pattern[str]) -> dict[str, Any]:
    rows = load_jsonl(path)
    target_hits: list[dict[str, Any]] = []
    top1_counts: Counter[str] = Counter()
    for row in rows:
        recommendations = row.get("recommendations", [])
        if recommendations:
            top1_counts[str(recommendations[0].get("label", ""))] += 1
        for item in recommendations:
            if recommendation_matches_target(item, pattern):
                target_hits.append(
                    {
                        "rank": int(item.get("rank", 0)),
                        "label": item.get("label"),
                        "score": float(item.get("score", 0.0)),
                        "function_name": row.get("function_name"),
                        "region_index": row.get("region_index"),
                        "change_probability": float(row.get("change_probability", 0.0)),
                    }
                )
                break
    best_hit = None
    if target_hits:
        best_hit = sorted(target_hits, key=lambda hit: (hit["rank"], -hit["score"]))[0]
    return {
        "regions": len(rows),
        "target_hit_regions": len(target_hits),
        "best_target_hit": best_hit,
        "top1_label_counts": top1_counts.most_common(10),
    }


def run_probe(args: argparse.Namespace, root: Path, gemm_ir: Path, round_index: int) -> dict[str, Any]:
    epoch = latest_epoch(args.model_dir)
    checkpoint = args.model_dir / f"{args.probe_checkpoint}.pt"
    if not checkpoint.exists():
        raise SystemExit(f"missing probe checkpoint: {checkpoint}")
    stem = f"round_{round_index:02d}"
    if epoch is not None:
        stem += f"_epoch_{epoch}"
    out_jsonl = args.probe_out_dir / f"{stem}.jsonl"
    out_md = args.probe_out_dir / f"{stem}.md"
    cmd = [
        sys.executable,
        "scripts/recommend_region_intrinsics.py",
        "--ir-file",
        str(gemm_ir),
        "--checkpoint",
        str(checkpoint),
        "--instruction-graph-dir",
        str(args.probe_instruction_graph_dir),
        "--out-jsonl",
        str(out_jsonl),
        "--out-md",
        str(out_md),
        "--top-k",
        str(args.top_k),
        "--store-slice-depth",
        str(args.store_slice_depth),
        "--store-slice-max-nodes",
        str(args.store_slice_max_nodes),
        "--device",
        args.device,
    ]
    if args.only_predicted_change:
        cmd.append("--only-predicted-change")
    print(f"\n=== GEMM AMX probe round {round_index} ===", flush=True)
    run_command(cmd, root)

    pattern = re.compile(args.target_label_regex, re.IGNORECASE)
    summary = summarize_recommendations(out_jsonl, pattern)
    summary.update(
        {
            "round": round_index,
            "epoch": epoch,
            "checkpoint": str(checkpoint),
            "out_jsonl": str(out_jsonl),
            "out_md": str(out_md),
        }
    )
    if summary.get("best_target_hit"):
        snapshot_dir = args.probe_out_dir / "checkpoint_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_dir / f"{stem}_{args.probe_checkpoint}.pt"
        shutil.copy2(checkpoint, snapshot)
        summary["checkpoint_snapshot"] = str(snapshot)
    return summary


def write_history_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# GEMM AMX Probe History",
        "",
        "| round | epoch | hit regions | best rank | best label | best score | function | region | report |",
        "| ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        hit = row.get("best_target_hit") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("round", "")),
                    str(row.get("epoch", "")),
                    str(row.get("target_hit_regions", 0)),
                    str(hit.get("rank", "")),
                    f"`{hit.get('label', '')}`" if hit else "",
                    f"{float(hit.get('score', 0.0)):.4f}" if hit else "",
                    str(hit.get("function_name", "")),
                    str(hit.get("region_index", "")),
                    str(row.get("out_md", "")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_existing_probe_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_jsonl(path)


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Train or resume region-intrinsic training in chunks and run a GEMM AMX "
            "zero-shot recommendation probe after each chunk."
        )
    )
    parser.add_argument("--region-dir", type=Path, required=True)
    parser.add_argument("--train-instruction-graph-dir", type=Path, required=True)
    parser.add_argument("--probe-instruction-graph-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--train-config",
        type=Path,
        help=(
            "Optional config.json to reuse hyperparameters from an earlier run. "
            "If omitted, an existing model-dir/config.json is used when present; "
            "otherwise train_region_intrinsic_model.py defaults are used."
        ),
    )
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--epochs-per-round", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-checkpoint", choices=["last", "best"], default="last")
    parser.add_argument("--probe-out-dir", type=Path, default=Path("runs/gemm_recommendations/auto_probe"))
    parser.add_argument("--gemm-source", type=Path, default=Path("examples/gemm/gemm_scalar.c"))
    parser.add_argument("--gemm-ir", type=Path, default=Path("runs/gemm_prevec_ir/gemm_scalar.ll"))
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--gemm-opt-level", default="-O2")
    parser.add_argument("--clang-extra-arg", action="append", default=[])
    parser.add_argument("--skip-gemm-compile", action="store_true")
    parser.add_argument("--force-gemm-compile", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--store-slice-depth", type=int, default=8)
    parser.add_argument("--store-slice-max-nodes", type=int, default=256)
    parser.add_argument("--only-predicted-change", action="store_true")
    parser.add_argument("--target-label-regex", default=r"amx|tdpb|tile")
    parser.add_argument("--allow-target-in-train-graphs", action="store_true")
    parser.add_argument("--allow-no-target-in-probe-graphs", action="store_true")
    parser.add_argument(
        "--snapshot-on-target",
        action="store_true",
        help="Deprecated; target-hit checkpoints are now always snapshotted.",
    )
    args = parser.parse_args()

    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.epochs_per_round <= 0:
        parser.error("--epochs-per-round must be positive")

    for attr in ("region_dir", "train_instruction_graph_dir", "probe_instruction_graph_dir", "model_dir"):
        path = getattr(args, attr)
        if not path.is_absolute():
            setattr(args, attr, root / path)
    if args.train_config is not None and not args.train_config.is_absolute():
        args.train_config = root / args.train_config
    if not args.probe_out_dir.is_absolute():
        args.probe_out_dir = root / args.probe_out_dir
    args.probe_out_dir.mkdir(parents=True, exist_ok=True)

    target_pattern = re.compile(args.target_label_regex, re.IGNORECASE)
    train_target_count = count_matching_labels(args.train_instruction_graph_dir, target_pattern)
    probe_target_count = count_matching_labels(args.probe_instruction_graph_dir, target_pattern)
    if train_target_count and not args.allow_target_in_train_graphs:
        raise SystemExit(
            "training instruction graph directory appears to contain target labels "
            f"({train_target_count} matches). Use a no-AMX training directory, or pass "
            "--allow-target-in-train-graphs intentionally."
        )
    if not probe_target_count and not args.allow_no_target_in_probe_graphs:
        raise SystemExit(
            "probe instruction graph directory has no target labels. Build it with AMX specs first."
        )
    print(
        json.dumps(
            {
                "train_target_label_matches": train_target_count,
                "probe_target_label_matches": probe_target_count,
                "target_label_regex": args.target_label_regex,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    gemm_ir = compile_gemm_ir(args, root)
    history_path = args.probe_out_dir / "probe_history.jsonl"
    history = load_existing_probe_history(history_path)
    start_round = len(history) + 1
    for offset in range(args.rounds):
        round_index = start_round + offset
        train_one_round(args, root, round_index)
        summary = run_probe(args, root, gemm_ir, round_index)
        history.append(summary)
        write_jsonl(history_path, history)
        write_history_markdown(args.probe_out_dir / "probe_history.md", history)
        print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
