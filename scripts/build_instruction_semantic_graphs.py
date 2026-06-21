#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

try:
    from mini_programl_graphs import (
        EDGE_TYPES,
        EDGE_FEATURE_TYPES,
        NODE_TYPES,
        as_list,
        build_vocab,
        compact_graph_features,
        compact_node_features,
        encode_graph,
        first_text,
        instruction_label,
        load_jsonl,
        parse_intrinsic_spec_to_graph,
        record_aliases,
        write_jsonl,
    )
except ModuleNotFoundError:  # pragma: no cover - used when imported as a package module
    from scripts.mini_programl_graphs import (
        EDGE_TYPES,
        EDGE_FEATURE_TYPES,
        NODE_TYPES,
        as_list,
        build_vocab,
        compact_graph_features,
        compact_node_features,
        encode_graph,
        first_text,
        instruction_label,
        load_jsonl,
        parse_intrinsic_spec_to_graph,
        record_aliases,
        write_jsonl,
    )


def normalized_extension(record: dict[str, Any]) -> str:
    return first_text(record, "extension", "instruction_set").upper().replace("-", "_")


def normalized_architecture(record: dict[str, Any]) -> str:
    return first_text(record, "architecture", "isa").lower()


def load_split_policy(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def policy_list(policy: dict[str, Any], key: str) -> set[str]:
    return {str(item).upper().replace("-", "_") for item in policy.get(key, [])}


def policy_pairs(policy: dict[str, Any], key: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for item in policy.get(key, []):
        text = str(item)
        if ":" in text:
            arch, extension = text.split(":", 1)
            pairs.add((arch.lower(), extension.upper().replace("-", "_")))
        else:
            pairs.add(("", text.upper().replace("-", "_")))
    return pairs


def assign_split(record: dict[str, Any], policy: dict[str, Any]) -> str:
    if not policy:
        return str(record.get("split", "train"))
    arch = normalized_architecture(record)
    extension = normalized_extension(record)
    pair = (arch, extension)
    train_pairs = policy_pairs(policy, "train_extensions")
    test_pairs = policy_pairs(policy, "test_extensions")
    train_extensions = policy_list(policy, "train_extension_names")
    test_extensions = policy_list(policy, "test_extension_names")
    if pair in train_pairs or extension in train_extensions:
        return "train"
    if pair in test_pairs or extension in test_extensions:
        return "test"
    if arch == "x86" and policy.get("x86_test_all_except_train", False):
        return "test"
    if arch == "riscv" and policy.get("riscv_default_split") == "test":
        return "test"
    return str(record.get("split", "train"))


def make_output_record(
    record: dict[str, Any],
    symbolic: dict[str, Any],
    token_to_id: dict[str, int],
    include_symbols: bool,
    include_debug_text: bool,
    split: str,
) -> dict[str, Any]:
    encoded = encode_graph(symbolic, token_to_id)
    output = {
        "label": instruction_label(record),
        "split": split,
        "architecture": first_text(record, "architecture", "isa"),
        "extension": first_text(record, "extension", "instruction_set"),
        "intrinsic": first_text(record, "llvm_intrinsic", "intrinsic"),
        "instruction": first_text(record, "instruction", "assembly"),
        "aliases": record_aliases(record),
        "graph_schema": symbolic.get("graph_schema", "mini-programl"),
        "graph_kind": symbolic.get("graph_kind", "intrinsic"),
        "num_nodes": len(encoded["node_token_ids"]),
        "num_edges": len(encoded["edges"]),
        "node_token_ids": encoded["node_token_ids"],
        "node_type_ids": encoded["node_type_ids"],
        "edges": encoded["edges"],
        "edge_feature_ids": encoded["edge_feature_ids"],
        "edge_counts": encoded["edge_counts"],
    }
    if include_symbols:
        output["node_tokens"] = symbolic["node_tokens"]
        output["node_types"] = symbolic["node_types"]
        output["node_features"] = compact_node_features(symbolic["node_features"], include_debug_text)
        output["symbolic_edges"] = symbolic["edges"]
        output["edge_features"] = symbolic["edge_features"]
        output["graph_features"] = compact_graph_features(symbolic["graph_features"], include_debug_text)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build mini-ProGraML instruction graphs from intrinsic/instruction Operation specs."
    )
    parser.add_argument("--spec-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/instruction_semantics/sample"))
    parser.add_argument("--split-policy", type=Path)
    parser.add_argument("--max-token-vocab", type=int, default=20000)
    parser.add_argument("--min-token-count", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=512)
    parser.add_argument("--max-edges", type=int, default=4096)
    parser.add_argument("--max-operand-nodes", type=int, default=256)
    parser.add_argument("--drop-symbols", action="store_true", help="Do not store node tokens/types in graph rows.")
    parser.add_argument("--include-debug-text", action="store_true", help="Keep debug-only full_text fields in symbolic output.")
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for path in args.spec_jsonl:
        records.extend(load_jsonl(path))
    if not records:
        raise SystemExit("no instruction specs found")
    split_policy = load_split_policy(args.split_policy)

    symbolic_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    token_counter: Counter[str] = Counter()
    for record in records:
        symbolic = parse_intrinsic_spec_to_graph(
            record,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            max_operand_nodes=args.max_operand_nodes,
        )
        symbolic_rows.append((record, symbolic))
        token_counter.update(symbolic["node_tokens"])

    token_to_id = build_vocab(token_counter, args.max_token_vocab, args.min_token_count)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vocab = {
        "token_to_id": token_to_id,
        "node_type_to_id": NODE_TYPES,
        "edge_type_to_id": EDGE_TYPES,
        "edge_feature_to_id": EDGE_FEATURE_TYPES,
        "params": {
            "graph_schema": "mini-programl",
            "graph_kind": "intrinsic",
            "max_token_vocab": args.max_token_vocab,
            "min_token_count": args.min_token_count,
            "max_nodes": args.max_nodes,
            "max_edges": args.max_edges,
            "max_operand_nodes": args.max_operand_nodes,
        },
    }
    (args.out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    summary: dict[str, Any] = {"splits": {}, "vocab": str(args.out_dir / "vocab.json")}
    for record, symbolic in symbolic_rows:
        split = assign_split(record, split_policy)
        by_split[split].append(
            make_output_record(
                record,
                symbolic,
                token_to_id,
                include_symbols=not args.drop_symbols,
                include_debug_text=args.include_debug_text,
                split=split,
            )
        )

    for split, rows in sorted(by_split.items()):
        out_path = args.out_dir / f"{split}.instruction_graphs.jsonl"
        write_jsonl(out_path, rows)
        summary["splits"][split] = {
            "rows": len(rows),
            "path": str(out_path),
            "architectures": sorted({row["architecture"] for row in rows if row["architecture"]}),
            "extensions": sorted({row["extension"] for row in rows if row["extension"]}),
        }
        print(json.dumps({"split": split, **summary["splits"][split]}, sort_keys=True))

    summary["token_vocab_size"] = len(token_to_id)
    if args.split_policy:
        summary["split_policy"] = str(args.split_policy)
    summary["node_type_to_id"] = NODE_TYPES
    summary["edge_type_to_id"] = EDGE_TYPES
    summary["edge_feature_to_id"] = EDGE_FEATURE_TYPES
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
