#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

try:
    from mini_programl_graphs import (
        EDGE_TYPES,
        EDGE_FEATURE_TYPES,
        NODE_TYPES,
        build_vocab,
        encode_graph,
        load_jsonl,
        parse_program_ir_to_graph,
        write_jsonl,
    )
except ModuleNotFoundError:  # pragma: no cover - used when imported as a package module
    from scripts.mini_programl_graphs import (
        EDGE_TYPES,
        EDGE_FEATURE_TYPES,
        NODE_TYPES,
        build_vocab,
        encode_graph,
        load_jsonl,
        parse_program_ir_to_graph,
        write_jsonl,
    )


def split_path(split_dir: Path, split: str) -> Path:
    return split_dir / f"{split}.jsonl"


def label_path(label_dir: Path, split: str) -> Path:
    return label_dir / f"{split}.labels.jsonl"


def record_key(row: dict[str, Any]) -> str:
    return str(row.get("dataset_index"))


def instruction_graph_paths(graph_dir: Path) -> list[Path]:
    return sorted(graph_dir.glob("*.instruction_graphs.jsonl"))


def load_instruction_graph_labels(graph_dir: Path, split: str | None = None) -> set[str]:
    labels: set[str] = set()
    for path in instruction_graph_paths(graph_dir):
        if split is not None and path.name != f"{split}.instruction_graphs.jsonl":
            continue
        for row in load_jsonl(path):
            if row.get("label"):
                labels.add(str(row["label"]))
            labels.update(str(item) for item in row.get("aliases", []) if str(item))
    return labels


def ir_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None and isinstance(row.get("ir"), dict):
        value = row["ir"].get(field)
    return str(value or "")


def load_joined(split_dir: Path, label_dir: Path, split: str, ir_field: str) -> list[dict[str, Any]]:
    source_rows = {record_key(row): row for row in load_jsonl(split_path(split_dir, split))}
    label_rows = load_jsonl(label_path(label_dir, split))
    rows: list[dict[str, Any]] = []
    for label in label_rows:
        if not label.get("ok"):
            continue
        key = record_key(label)
        source = source_rows.get(key)
        if source is None:
            continue
        instructions = [str(item) for item in label.get("instructions", [])]
        if not instructions:
            continue
        rows.append(
            {
                "dataset_index": key,
                "repo_name": source.get("repo_name"),
                "repo_file_path": source.get("repo_file_path"),
                "function_name": source.get("function_name"),
                "structural_hash": source.get("structural_hash"),
                "ir": ir_text(source, ir_field),
                "instructions": instructions,
            }
        )
    return rows


def build_program_graph(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return parse_program_ir_to_graph(
        row["ir"],
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        max_operand_nodes=args.max_operand_nodes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build mini-ProGraML program graph JSONL records from IR-OptSet labels."
    )
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--instruction-graph-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/iroptset/graphs/pilot_10k_v1"))
    parser.add_argument("--ir-field", default="preprocessed_ir")
    parser.add_argument("--max-token-vocab", type=int, default=20000)
    parser.add_argument("--min-token-count", type=int, default=2)
    parser.add_argument("--max-instruction-vocab", type=int, default=512)
    parser.add_argument("--min-instruction-count", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=512)
    parser.add_argument("--max-edges", type=int, default=4096)
    parser.add_argument("--max-operand-nodes", type=int, default=128)
    parser.add_argument(
        "--max-train-examples-per-instruction",
        type=int,
        default=0,
        help="Greedy cap for train rows per instruction label; 0 disables balancing.",
    )
    args = parser.parse_args()

    train_rows = load_joined(args.split_dir, args.label_dir, "train", args.ir_field)
    if not train_rows:
        raise SystemExit("no joined train rows found")

    allowed_train_labels: set[str] = set()
    all_instruction_labels: set[str] = set()
    if args.instruction_graph_dir:
        all_instruction_labels = load_instruction_graph_labels(args.instruction_graph_dir)
        allowed_train_labels = load_instruction_graph_labels(args.instruction_graph_dir, split="train")
        if not all_instruction_labels:
            raise SystemExit(f"no instruction graph labels found in {args.instruction_graph_dir}")
        if not allowed_train_labels:
            raise SystemExit(f"no split=train instruction graph labels found in {args.instruction_graph_dir}")

    token_counter: Counter[str] = Counter()
    instruction_counter: Counter[str] = Counter()
    train_symbolic: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        if allowed_train_labels:
            row["instructions"] = [item for item in row["instructions"] if item in allowed_train_labels]
            if not row["instructions"]:
                continue
        graph = build_program_graph(row, args)
        train_symbolic[row["dataset_index"]] = graph
        token_counter.update(graph["node_tokens"])
        instruction_counter.update(row["instructions"])

    token_to_id = build_vocab(token_counter, args.max_token_vocab, args.min_token_count)
    if all_instruction_labels:
        instruction_to_id = {instruction: index for index, instruction in enumerate(sorted(all_instruction_labels))}
    else:
        instruction_to_id = build_vocab(
            instruction_counter, args.max_instruction_vocab + 2, args.min_instruction_count
        )
        instruction_to_id.pop("<pad>", None)
        instruction_to_id.pop("<unk>", None)
        instruction_to_id = {instruction: index for index, instruction in enumerate(instruction_to_id)}
    if not instruction_to_id:
        raise SystemExit("instruction vocabulary is empty")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vocab = {
        "token_to_id": token_to_id,
        "node_type_to_id": NODE_TYPES,
        "edge_type_to_id": EDGE_TYPES,
        "edge_feature_to_id": EDGE_FEATURE_TYPES,
        "instruction_to_id": instruction_to_id,
        "params": {
            "graph_schema": "mini-programl",
            "graph_kind": "program",
            "ir_field": args.ir_field,
            "max_nodes": args.max_nodes,
            "max_edges": args.max_edges,
            "max_operand_nodes": args.max_operand_nodes,
            "max_token_vocab": args.max_token_vocab,
            "min_token_count": args.min_token_count,
            "max_instruction_vocab": args.max_instruction_vocab,
            "min_instruction_count": args.min_instruction_count,
            "instruction_graph_dir": str(args.instruction_graph_dir) if args.instruction_graph_dir else None,
            "max_train_examples_per_instruction": args.max_train_examples_per_instruction,
        },
    }
    (args.out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    summary: dict[str, Any] = {"splits": {}, "vocab": str(args.out_dir / "vocab.json")}
    for split in ("train", "val", "test"):
        if not split_path(args.split_dir, split).exists() or not label_path(args.label_dir, split).exists():
            continue
        rows = load_joined(args.split_dir, args.label_dir, split, args.ir_field)
        output_rows: list[dict[str, Any]] = []
        skipped_no_label = 0
        skipped_balance = 0
        train_label_counts: Counter[int] = Counter()
        for row in rows:
            graph = train_symbolic.get(row["dataset_index"])
            if graph is None:
                graph = build_program_graph(row, args)
            instructions = [item for item in row["instructions"] if item in instruction_to_id]
            if split == "train" and allowed_train_labels:
                instructions = [item for item in instructions if item in allowed_train_labels]
            label_ids = sorted({instruction_to_id[item] for item in instructions})
            if not label_ids:
                skipped_no_label += 1
                continue
            if (
                split == "train"
                and args.max_train_examples_per_instruction > 0
                and all(train_label_counts[label_id] >= args.max_train_examples_per_instruction for label_id in label_ids)
            ):
                skipped_balance += 1
                continue
            if split == "train" and args.max_train_examples_per_instruction > 0:
                for label_id in label_ids:
                    train_label_counts[label_id] += 1
            encoded = encode_graph(graph, token_to_id)
            output_rows.append(
                {
                    "dataset_index": row["dataset_index"],
                    "repo_name": row.get("repo_name"),
                    "repo_file_path": row.get("repo_file_path"),
                    "function_name": row.get("function_name"),
                    "structural_hash": row.get("structural_hash"),
                    "graph_schema": graph.get("graph_schema", "mini-programl"),
                    "graph_kind": graph.get("graph_kind", "program"),
                    "num_nodes": len(encoded["node_token_ids"]),
                    "num_edges": len(encoded["edges"]),
                    "node_token_ids": encoded["node_token_ids"],
                    "node_type_ids": encoded["node_type_ids"],
                    "edges": encoded["edges"],
                    "edge_feature_ids": encoded["edge_feature_ids"],
                    "edge_counts": encoded["edge_counts"],
                    "label_ids": label_ids,
                    "instructions": instructions,
                }
            )
        out_path = args.out_dir / f"{split}.graphs.jsonl"
        write_jsonl(out_path, output_rows)
        summary["splits"][split] = {
            "rows": len(output_rows),
            "skipped_no_label": skipped_no_label,
            "skipped_balance": skipped_balance,
            "path": str(out_path),
        }
        print(json.dumps({"split": split, **summary["splits"][split]}, sort_keys=True))

    summary["token_vocab_size"] = len(token_to_id)
    summary["instruction_vocab_size"] = len(instruction_to_id)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
