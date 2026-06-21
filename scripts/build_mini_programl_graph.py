#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from mini_programl_graphs import (
        encode_graph,
        first_text,
        instruction_label,
        load_jsonl,
        compact_graph_features,
        compact_node_features,
        parse_mini_programl,
        record_aliases,
        write_jsonl,
    )
except ModuleNotFoundError:  # pragma: no cover - used when imported as a package module
    from scripts.mini_programl_graphs import (
        encode_graph,
        first_text,
        instruction_label,
        load_jsonl,
        compact_graph_features,
        compact_node_features,
        parse_mini_programl,
        record_aliases,
        write_jsonl,
    )


def load_vocab(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    token_to_id = data.get("token_to_id")
    if not isinstance(token_to_id, dict):
        raise SystemExit(f"{path} does not contain token_to_id")
    return {str(token): int(index) for token, index in token_to_id.items()}


def graph_output(
    symbolic: dict[str, Any],
    token_to_id: dict[str, int] | None,
    include_symbols: bool,
    include_debug_text: bool,
) -> dict[str, Any]:
    output = {
        "graph_schema": symbolic.get("graph_schema", "mini-programl"),
        "graph_kind": symbolic.get("graph_kind"),
        "num_nodes": len(symbolic["node_tokens"]),
        "num_edges": len(symbolic["edges"]),
    }
    if token_to_id is not None:
        output.update(encode_graph(symbolic, token_to_id))
    if include_symbols:
        output["node_tokens"] = symbolic["node_tokens"]
        output["node_types"] = symbolic["node_types"]
        output["node_features"] = compact_node_features(symbolic["node_features"], include_debug_text)
        output["symbolic_edges"] = symbolic["edges"]
        output["edge_features"] = symbolic["edge_features"]
        output["graph_features"] = compact_graph_features(symbolic.get("graph_features", {}), include_debug_text)
    return output


def program_record(args: argparse.Namespace, token_to_id: dict[str, int] | None) -> dict[str, Any]:
    if args.ir_file is None:
        raise SystemExit("--ir-file is required when --mode program")
    symbolic = parse_mini_programl(
        "program",
        args.ir_file.read_text(encoding="utf-8"),
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        max_operand_nodes=args.max_operand_nodes,
    )
    output = graph_output(
        symbolic,
        token_to_id,
        include_symbols=not args.drop_symbols,
        include_debug_text=args.include_debug_text,
    )
    output["ir_file"] = str(args.ir_file)
    return output


def intrinsic_records(args: argparse.Namespace, token_to_id: dict[str, int] | None) -> list[dict[str, Any]]:
    if not args.spec_jsonl:
        raise SystemExit("--spec-jsonl is required when --mode intrinsic")
    rows: list[dict[str, Any]] = []
    for path in args.spec_jsonl:
        for record in load_jsonl(path):
            symbolic = parse_mini_programl(
                "intrinsic",
                record,
                max_nodes=args.max_nodes,
                max_edges=args.max_edges,
                max_operand_nodes=args.max_operand_nodes,
            )
            output = graph_output(
                symbolic,
                token_to_id,
                include_symbols=not args.drop_symbols,
                include_debug_text=args.include_debug_text,
            )
            output.update(
                {
                    "label": instruction_label(record),
                    "architecture": first_text(record, "architecture", "isa"),
                    "extension": first_text(record, "extension", "instruction_set"),
                    "intrinsic": first_text(record, "llvm_intrinsic", "intrinsic"),
                    "instruction": first_text(record, "instruction", "assembly"),
                    "aliases": record_aliases(record),
                }
            )
            rows.append(output)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build one-off mini-ProGraML graphs for program IR or intrinsic instruction specs."
    )
    parser.add_argument("--mode", choices=["program", "intrinsic"], required=True)
    parser.add_argument("--ir-file", type=Path, help="LLVM IR file for --mode program.")
    parser.add_argument("--spec-jsonl", type=Path, nargs="+", help="Intrinsic spec JSONL for --mode intrinsic.")
    parser.add_argument("--vocab", type=Path, help="Optional vocab.json. When provided, emit encoded graph fields too.")
    parser.add_argument("--out", type=Path, help="Output JSON for program mode or JSONL for intrinsic mode.")
    parser.add_argument("--max-nodes", type=int, default=512)
    parser.add_argument("--max-edges", type=int, default=4096)
    parser.add_argument("--max-operand-nodes", type=int, default=256)
    parser.add_argument("--drop-symbols", action="store_true", help="Only emit encoded fields when --vocab is set.")
    parser.add_argument("--include-debug-text", action="store_true", help="Keep full_text and parsed AST fields.")
    args = parser.parse_args()

    token_to_id = load_vocab(args.vocab)
    if args.mode == "program":
        output = program_record(args, token_to_id)
        text = json.dumps(output, indent=2, sort_keys=True)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 0

    rows = intrinsic_records(args, token_to_id)
    if args.out:
        write_jsonl(args.out, rows)
    else:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
