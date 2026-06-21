#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from mini_programl_graphs import compact_graph_features, compact_node_features, parse_program_ir_to_graph
except ModuleNotFoundError:  # pragma: no cover - used when imported as a package module
    from scripts.mini_programl_graphs import compact_graph_features, compact_node_features, parse_program_ir_to_graph


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the symbolic program graph built from an LLVM IR file.")
    parser.add_argument("--ir-file", type=Path, required=True)
    parser.add_argument("--max-nodes", type=int, default=512)
    parser.add_argument("--max-edges", type=int, default=4096)
    parser.add_argument("--max-operand-nodes", type=int, default=128)
    parser.add_argument("--include-debug-text", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    graph = parse_program_ir_to_graph(
        args.ir_file.read_text(encoding="utf-8"),
        args.max_nodes,
        args.max_edges,
        args.max_operand_nodes,
    )
    graph["node_features"] = compact_node_features(graph["node_features"], args.include_debug_text)
    graph["graph_features"] = compact_graph_features(graph.get("graph_features", {}), args.include_debug_text)
    graph["ir_file"] = str(args.ir_file)
    text = json.dumps(graph, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
