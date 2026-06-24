#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

try:
    from build_llvm_vectorize_region_dataset import (
        candidate_regions,
        extract_defined_functions,
        parse_candidate_modes,
        stable_hash,
        subgraph_symbolic,
    )
    from mini_programl_graphs import (
        encode_graph,
        first_text,
        instruction_label,
        load_jsonl,
        parse_intrinsic_spec_to_graph,
        parse_program_ir_to_graph,
        record_aliases,
    )
    from train_instruction_dual_gnn import RelationalGraphEncoder, batch_graphs, encode_all
    from train_region_intrinsic_model import (
        load_instruction_graphs,
        matching_logits,
        require_torch,
    )
except ModuleNotFoundError:  # pragma: no cover - used when imported as a package module
    from scripts.build_llvm_vectorize_region_dataset import (
        candidate_regions,
        extract_defined_functions,
        parse_candidate_modes,
        stable_hash,
        subgraph_symbolic,
    )
    from scripts.mini_programl_graphs import (
        encode_graph,
        first_text,
        instruction_label,
        load_jsonl,
        parse_intrinsic_spec_to_graph,
        parse_program_ir_to_graph,
        record_aliases,
    )
    from scripts.train_instruction_dual_gnn import RelationalGraphEncoder, batch_graphs, encode_all
    from scripts.train_region_intrinsic_model import (
        load_instruction_graphs,
        matching_logits,
        require_torch,
    )


def discover_ir_files(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    files.extend(args.ir_file or [])
    for directory in args.ir_dir or []:
        files.extend(sorted(directory.rglob(args.glob)))
    unique: dict[str, Path] = {}
    for path in files:
        if path.exists():
            unique[str(path.resolve())] = path
    return sorted(unique.values(), key=lambda item: str(item))


def make_region_record(
    source_file: Path,
    function_name: str,
    region_index: int,
    symbolic: dict[str, Any],
    token_to_id: dict[str, int],
    include_symbols: bool,
) -> dict[str, Any]:
    encoded = encode_graph(symbolic, token_to_id)
    record: dict[str, Any] = {
        "region_id": stable_hash(
            f"{source_file}:{function_name}:{region_index}:{symbolic.get('graph_features', {}).get('source_nodes', [])}"
        )[:16],
        "source_file": str(source_file),
        "function_name": function_name,
        "region_index": region_index,
        "graph_schema": symbolic.get("graph_schema", "mini-programl"),
        "graph_kind": symbolic.get("graph_kind", "changed_region"),
        "num_nodes": len(symbolic["node_tokens"]),
        "num_edges": len(symbolic["edges"]),
        "node_token_ids": encoded["node_token_ids"],
        "node_type_ids": encoded["node_type_ids"],
        "edges": encoded["edges"],
        "edge_feature_ids": encoded["edge_feature_ids"],
        "edge_counts": encoded["edge_counts"],
        "source_nodes": symbolic.get("graph_features", {}).get("source_nodes", []),
    }
    if include_symbols:
        record["node_tokens"] = symbolic["node_tokens"]
        record["node_types"] = symbolic["node_types"]
        record["node_features"] = symbolic["node_features"]
        record["symbolic_edges"] = symbolic["edges"]
        record["edge_features"] = symbolic["edge_features"]
    return record


def build_regions_for_ir(
    ir_file: Path,
    token_to_id: dict[str, int],
    include_symbols: bool,
    max_regions_per_function: int,
    candidate_modes: tuple[str, ...],
    store_slice_depth: int,
    store_slice_max_nodes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    ir = ir_file.read_text(encoding="utf-8", errors="replace")
    functions = extract_defined_functions(ir)
    if not functions:
        functions = {ir_file.stem: ir}
    for function_name, function_ir in functions.items():
        try:
            graph = parse_program_ir_to_graph(function_ir)
            candidates = candidate_regions(
                graph,
                modes=candidate_modes,
                store_slice_depth=store_slice_depth,
                store_slice_max_nodes=store_slice_max_nodes,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue per input file
            failures.append(
                {
                    "source_file": str(ir_file),
                    "function_name": function_name,
                    "error": str(exc),
                }
            )
            continue
        if max_regions_per_function > 0:
            candidates = candidates[:max_regions_per_function]
        for region_index, selected in enumerate(candidates):
            symbolic = subgraph_symbolic(graph, selected)
            rows.append(
                make_region_record(
                    ir_file,
                    function_name,
                    region_index,
                    symbolic,
                    token_to_id,
                    include_symbols,
                )
            )
    return rows, failures


def load_checkpoint(torch, checkpoint_path: Path, device: str) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    for key in ("region_encoder_state", "instruction_encoder_state", "change_head_state", "config"):
        if key not in checkpoint:
            raise SystemExit(f"checkpoint is missing {key}: {checkpoint_path}")
    return checkpoint


def load_instruction_vocab(instruction_graph_dir: Path) -> dict[str, Any] | None:
    vocab_path = instruction_graph_dir / "vocab.json"
    if not vocab_path.exists():
        return None
    return json.loads(vocab_path.read_text(encoding="utf-8"))


def instruction_vocab_mismatches(checkpoint_vocab: dict[str, Any], graph_vocab: dict[str, Any] | None) -> list[str]:
    if graph_vocab is None:
        return ["missing_vocab_json"]
    keys = ("token_to_id", "node_type_to_id", "edge_type_to_id", "edge_feature_to_id")
    return [key for key in keys if checkpoint_vocab.get(key) != graph_vocab.get(key)]


def has_symbolic_instruction_graph(row: dict[str, Any]) -> bool:
    return all(key in row for key in ("node_tokens", "node_types", "symbolic_edges", "edge_features"))


def reencode_symbolic_instruction_row(row: dict[str, Any], token_to_id: dict[str, int]) -> dict[str, Any]:
    symbolic = {
        "node_tokens": row["node_tokens"],
        "node_types": row["node_types"],
        "edges": row["symbolic_edges"],
        "edge_features": row["edge_features"],
    }
    encoded = encode_graph(symbolic, token_to_id)
    output = dict(row)
    output.update(
        {
            "node_token_ids": encoded["node_token_ids"],
            "node_type_ids": encoded["node_type_ids"],
            "edges": encoded["edges"],
            "edge_feature_ids": encoded["edge_feature_ids"],
            "edge_counts": encoded["edge_counts"],
        }
    )
    return output


def reencode_instruction_rows(
    rows: list[dict[str, Any]],
    checkpoint_vocab: dict[str, Any],
    graph_vocab: dict[str, Any] | None,
    instruction_graph_dir: Path,
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    mismatched = instruction_vocab_mismatches(checkpoint_vocab, graph_vocab)
    token_to_id = checkpoint_vocab["token_to_id"]
    output: list[dict[str, Any]] = []
    missing_symbolic = 0
    for row in rows:
        if has_symbolic_instruction_graph(row):
            output.append(reencode_symbolic_instruction_row(row, token_to_id))
        else:
            missing_symbolic += 1
            output.append(row)
    if mismatched and missing_symbolic:
        joined = ", ".join(mismatched)
        raise SystemExit(
            "instruction graph vocab does not match checkpoint and some graph rows "
            "lack symbolic node/edge fields needed for re-encoding "
            f"({joined}). Rebuild the instruction graphs without --drop-symbols, "
            f"or use the original training graph directory. checkpoint={checkpoint_path} "
            f"instruction_graph_dir={instruction_graph_dir}"
        )
    return output


def make_instruction_record(
    record: dict[str, Any],
    symbolic: dict[str, Any],
    token_to_id: dict[str, int],
    split: str,
) -> dict[str, Any]:
    encoded = encode_graph(symbolic, token_to_id)
    return {
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
        "node_tokens": symbolic["node_tokens"],
        "node_types": symbolic["node_types"],
        "node_features": symbolic["node_features"],
        "symbolic_edges": symbolic["edges"],
        "edge_features": symbolic["edge_features"],
        "graph_features": symbolic["graph_features"],
    }


def load_candidate_instruction_specs(
    paths: list[Path],
    token_to_id: dict[str, int],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in paths:
        for record in load_jsonl(path):
            candidate = dict(record)
            candidate["split"] = "candidate"
            symbolic = parse_intrinsic_spec_to_graph(candidate, root_token_mode="generic")
            candidates.append(make_instruction_record(candidate, symbolic, token_to_id, "candidate"))
    return candidates


def make_encoder(torch, nn, F, vocab: dict[str, Any], config: dict[str, Any], device: str):
    return RelationalGraphEncoder(
        torch,
        nn,
        F,
        token_vocab_size=len(vocab["token_to_id"]),
        node_type_count=len(vocab["node_type_to_id"]),
        edge_type_count=len(vocab["edge_type_to_id"]),
        edge_feature_count=len(vocab.get("edge_feature_to_id", {"none": 0})),
        max_edge_position=int(config.get("max_edge_position", 32)),
        max_edge_count=int(config.get("max_edge_count", 128)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        layers=int(config.get("layers", 3)),
        dropout=float(config.get("dropout", 0.15)),
    ).model.to(device)


def recommendation_entry(row: dict[str, Any], score: float, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "label": row.get("label"),
        "score": score,
        "instruction": row.get("instruction"),
        "intrinsic": row.get("intrinsic"),
        "extension": row.get("extension"),
        "architecture": row.get("architecture"),
    }


DEFAULT_EXCLUDE_RECOMMENDATION_LABEL_REGEXES = (
    r"^llvm\.vector\.(?:load|store)\.",
)


def compile_label_regexes(patterns: list[str] | tuple[str, ...]) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns if pattern]


def keep_recommendation_label(row: dict[str, Any], exclude_regexes: list[re.Pattern[str]]) -> bool:
    label = str(row.get("label", ""))
    return not any(pattern.search(label) for pattern in exclude_regexes)


def predict(
    torch,
    F,
    region_encoder,
    instruction_encoder,
    change_head,
    regions: list[dict[str, Any]],
    instruction_rows: list[dict[str, Any]],
    batch_size: int,
    instruction_batch_size: int,
    top_k: int,
    temperature: float,
    device: str,
) -> list[dict[str, Any]]:
    region_encoder.eval()
    instruction_encoder.eval()
    change_head.eval()
    output: list[dict[str, Any]] = []
    top_k = min(top_k, len(instruction_rows))
    with torch.no_grad():
        instruction_emb = encode_all(torch, instruction_encoder, instruction_rows, instruction_batch_size, device)
        for start in range(0, len(regions), batch_size):
            selected = regions[start : start + batch_size]
            batch = batch_graphs(torch, selected, 0, len(selected), device)
            region_emb = region_encoder(batch)
            change_logits = change_head(region_emb).squeeze(-1)
            change_probs = torch.sigmoid(change_logits)
            logits = matching_logits(torch, F, region_emb, instruction_emb, temperature)
            top_scores, top_indices = torch.topk(logits, k=top_k, dim=1)
            for row_index, row in enumerate(selected):
                enriched = dict(row)
                enriched["change_logit"] = float(change_logits[row_index].item())
                enriched["change_probability"] = float(change_probs[row_index].item())
                enriched["recommendations"] = [
                    recommendation_entry(
                        instruction_rows[int(label_index)],
                        float(top_scores[row_index, rank].item()),
                        rank + 1,
                    )
                    for rank, label_index in enumerate(top_indices[row_index].tolist())
                ]
                output.append(enriched)
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def write_markdown(path: Path, rows: list[dict[str, Any]], max_rows: int, top_k: int) -> None:
    selected = sorted(rows, key=lambda row: float(row.get("change_probability", 0.0)), reverse=True)[:max_rows]
    lines = ["# Region Intrinsic Recommendations", ""]
    lines.append(f"Showing top {len(selected)} regions by change probability.")
    lines.append("")
    for row in selected:
        lines.append(
            "## "
            + f"{Path(str(row.get('source_file'))).name} :: {row.get('function_name')} "
            + f"region {row.get('region_index')}"
        )
        lines.append("")
        lines.append(
            f"- change_probability: `{float(row.get('change_probability', 0.0)):.4f}`"
        )
        lines.append(f"- nodes: `{row.get('num_nodes')}`, edges: `{row.get('num_edges')}`")
        if row.get("node_tokens"):
            tokens = " ".join(str(token) for token in row["node_tokens"][:48])
            lines.append(f"- region_tokens: `{tokens}`")
        lines.append("")
        lines.append("| rank | label | score |")
        lines.append("| ---: | --- | ---: |")
        for item in row.get("recommendations", [])[:top_k]:
            lines.append(f"| {item['rank']} | `{item['label']}` | {float(item['score']):.4f} |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recommend intrinsic/vector labels for pre-vectorization LLVM IR regions."
    )
    parser.add_argument("--ir-file", type=Path, action="append")
    parser.add_argument("--ir-dir", type=Path, action="append")
    parser.add_argument("--glob", default="*.ll")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--instruction-graph-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-spec-jsonl",
        type=Path,
        nargs="*",
        default=[],
        help=(
            "Additional zero-shot candidate instruction specs to append at inference time. "
            "They are encoded with the checkpoint instruction vocabulary and are not part of training."
        ),
    )
    parser.add_argument("--out-jsonl", type=Path, default=Path("runs/llvm_vectorize_regions/recommendations.jsonl"))
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--include-memory-labels",
        action="store_true",
        help="Include generic llvm.vector.load/store labels in recommendation ranking.",
    )
    parser.add_argument(
        "--exclude-label-regex",
        action="append",
        default=[],
        help="Additional regex for labels to hide from recommendation ranking.",
    )
    parser.add_argument("--change-threshold", type=float, default=0.0)
    parser.add_argument("--only-predicted-change", action="store_true")
    parser.add_argument(
        "--candidate-modes",
        default="store-backward",
        help="Comma-separated region proposal modes: store-backward, loop-backedge, producer-consumer.",
    )
    parser.add_argument("--store-slice-depth", type=int, default=5)
    parser.add_argument("--store-slice-max-nodes", type=int, default=128)
    parser.add_argument("--max-regions-per-function", type=int, default=0)
    parser.add_argument("--max-report-regions", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--instruction-batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--drop-symbols", action="store_true")
    args = parser.parse_args()

    if not args.ir_file and not args.ir_dir:
        parser.error("pass at least one --ir-file or --ir-dir")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    try:
        candidate_modes = parse_candidate_modes(args.candidate_modes)
    except ValueError as exc:
        parser.error(str(exc))

    torch, nn, F = require_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    checkpoint = load_checkpoint(torch, args.checkpoint, device)
    config = dict(checkpoint["config"])
    temperature = float(config.get("temperature", 0.07))
    region_vocab = checkpoint.get("region_vocab")
    instruction_vocab = checkpoint.get("instruction_vocab")
    if not isinstance(region_vocab, dict) or not isinstance(instruction_vocab, dict):
        raise SystemExit("checkpoint must include region_vocab and instruction_vocab")

    instruction_rows = load_instruction_graphs(args.instruction_graph_dir)
    if not instruction_rows:
        raise SystemExit(f"no instruction graphs found in {args.instruction_graph_dir}")
    graph_instruction_vocab = load_instruction_vocab(args.instruction_graph_dir)
    instruction_rows = reencode_instruction_rows(
        instruction_rows,
        instruction_vocab,
        graph_instruction_vocab,
        args.instruction_graph_dir,
        args.checkpoint,
    )
    candidate_instruction_rows = load_candidate_instruction_specs(
        args.candidate_spec_jsonl,
        instruction_vocab["token_to_id"],
    )
    instruction_rows.extend(candidate_instruction_rows)
    exclude_label_patterns = [
        *(() if args.include_memory_labels else DEFAULT_EXCLUDE_RECOMMENDATION_LABEL_REGEXES),
        *args.exclude_label_regex,
    ]
    exclude_label_regexes = compile_label_regexes(exclude_label_patterns)
    ranking_instruction_rows = [
        row for row in instruction_rows if keep_recommendation_label(row, exclude_label_regexes)
    ]
    if not ranking_instruction_rows:
        raise SystemExit("no instruction graphs remain after recommendation label filtering")

    region_encoder = make_encoder(torch, nn, F, region_vocab, config, device)
    instruction_encoder = make_encoder(torch, nn, F, instruction_vocab, config, device)
    change_head = nn.Linear(int(config.get("hidden_dim", 128)), 1).to(device)
    region_encoder.load_state_dict(checkpoint["region_encoder_state"])
    instruction_encoder.load_state_dict(checkpoint["instruction_encoder_state"])
    change_head.load_state_dict(checkpoint["change_head_state"])

    ir_files = discover_ir_files(args)
    if not ir_files:
        raise SystemExit("no input LLVM IR files found")

    failures: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = []
    for ir_file in ir_files:
        file_regions, file_failures = build_regions_for_ir(
            ir_file,
            region_vocab["token_to_id"],
            include_symbols=not args.drop_symbols,
            max_regions_per_function=args.max_regions_per_function,
            candidate_modes=candidate_modes,
            store_slice_depth=args.store_slice_depth,
            store_slice_max_nodes=args.store_slice_max_nodes,
        )
        regions.extend(file_regions)
        failures.extend(file_failures)

    recommendations = predict(
        torch,
        F,
        region_encoder,
        instruction_encoder,
        change_head,
        regions,
        ranking_instruction_rows,
        args.batch_size,
        args.instruction_batch_size,
        args.top_k,
        temperature,
        device,
    )
    if args.only_predicted_change:
        recommendations = [
            row for row in recommendations if float(row["change_logit"]) > args.change_threshold
        ]

    write_jsonl(args.out_jsonl, recommendations)
    if args.out_md:
        write_markdown(args.out_md, recommendations, args.max_report_regions, args.top_k)

    top_label_counts = Counter(
        row["recommendations"][0]["label"] for row in recommendations if row.get("recommendations")
    )
    summary = {
        "ir_files": len(ir_files),
        "regions": len(regions),
        "written_regions": len(recommendations),
        "instruction_graphs": len(instruction_rows),
        "candidate_spec_graphs": len(candidate_instruction_rows),
        "candidate_spec_jsonl": [str(path) for path in args.candidate_spec_jsonl],
        "ranked_instruction_graphs": len(ranking_instruction_rows),
        "excluded_recommendation_label_regexes": exclude_label_patterns,
        "candidate_modes": list(candidate_modes),
        "store_slice_depth": args.store_slice_depth,
        "store_slice_max_nodes": args.store_slice_max_nodes,
        "checkpoint": str(args.checkpoint),
        "out_jsonl": str(args.out_jsonl),
        "out_md": str(args.out_md) if args.out_md else None,
        "top1_label_counts": top_label_counts.most_common(25),
        "failures": failures[:100],
        "failure_count": len(failures),
    }
    summary_path = args.out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
