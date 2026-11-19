#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
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
    from mini_programl_graphs import encode_graph, parse_program_ir_to_graph
    from graph_encoder import (
        RelationalGraphEncoder,
        batch_graphs,
        encode_all,
        instruction_graph_paths,
        load_jsonl,
    )
except ModuleNotFoundError:  # pragma: no cover - used when imported as a package module
    from scripts.build_llvm_vectorize_region_dataset import (
        candidate_regions,
        extract_defined_functions,
        parse_candidate_modes,
        stable_hash,
        subgraph_symbolic,
    )
    from scripts.mini_programl_graphs import encode_graph, parse_program_ir_to_graph
    from scripts.graph_encoder import (
        RelationalGraphEncoder,
        batch_graphs,
        encode_all,
        instruction_graph_paths,
        load_jsonl,
    )


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit(
            "missing PyTorch; install it on the training server, for example: "
            "python3 -m pip install torch"
        ) from exc
    return torch, nn, F


def set_seed(seed: int) -> None:
    random.seed(seed)


def region_path(region_dir: Path, split: str) -> Path:
    return region_dir / f"{split}.regions.jsonl"


def load_instruction_graphs(instruction_graph_dir: Path) -> list[dict[str, Any]]:
    paths = instruction_graph_paths(instruction_graph_dir)
    if not paths:
        raise SystemExit(f"no *.instruction_graphs.jsonl files found in {instruction_graph_dir}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_jsonl(path))
    return rows


def instruction_key_map(instruction_rows: list[dict[str, Any]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, row in enumerate(instruction_rows):
        keys = [str(row["label"])]
        keys.extend(str(item) for item in row.get("aliases", []) if str(item))
        keys.extend(str(item) for item in row.get("intrinsic", "").split() if str(item))
        for key in keys:
            mapping.setdefault(key, index)
    return mapping


def align_region_rows(rows: list[dict[str, Any]], instruction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    label_to_id = instruction_key_map(instruction_rows)
    aligned: list[dict[str, Any]] = []
    for row in rows:
        ids = sorted(
            {
                label_to_id[label]
                for label in (str(item) for item in row.get("target_labels", []))
                if label in label_to_id
            }
        )
        aligned_row = dict(row)
        aligned_row["target_label_ids"] = ids
        aligned.append(aligned_row)
    return aligned


def label_matrix(torch, rows: list[dict[str, Any]], num_labels: int, device: str):
    labels = torch.zeros((len(rows), num_labels), dtype=torch.float32, device=device)
    for row_index, row in enumerate(rows):
        for label_id in row.get("target_label_ids", []):
            if 0 <= int(label_id) < num_labels:
                labels[row_index, int(label_id)] = 1.0
    return labels


def change_labels(torch, rows: list[dict[str, Any]], device: str):
    return torch.tensor([float(row.get("change_label", 0)) for row in rows], dtype=torch.float32, device=device)


def match_positive_weight(torch, rows: list[dict[str, Any]], num_labels: int, max_weight: float, device: str):
    positives = torch.zeros((num_labels,), dtype=torch.float32)
    total = 0
    for row in rows:
        if int(row.get("change_label", 0)) <= 0:
            continue
        total += 1
        for label_id in row.get("target_label_ids", []):
            positives[int(label_id)] += 1.0
    negatives = max(1, total) - positives
    weights = negatives / positives.clamp_min(1.0)
    return torch.clamp(weights, min=1.0, max=max_weight).to(device)


def change_positive_weight(torch, rows: list[dict[str, Any]], max_weight: float, device: str):
    positives = sum(1 for row in rows if int(row.get("change_label", 0)) > 0)
    negatives = max(0, len(rows) - positives)
    weight = negatives / max(1, positives)
    return torch.tensor(min(max(1.0, weight), max_weight), dtype=torch.float32, device=device)


def matching_logits(torch, F, region_emb, instruction_emb, temperature: float):
    return F.normalize(region_emb, dim=1) @ F.normalize(instruction_emb, dim=1).T / temperature


def batch_loss(
    torch,
    F,
    region_encoder,
    instruction_encoder,
    change_head,
    rows: list[dict[str, Any]],
    instruction_rows: list[dict[str, Any]],
    instruction_batch_size: int,
    change_criterion,
    match_criterion,
    temperature: float,
    match_loss_weight: float,
    device: str,
):
    batch = batch_graphs(torch, rows, 0, len(rows), device)
    region_emb = region_encoder(batch)
    change_logits = change_head(region_emb).squeeze(-1)
    changes = change_labels(torch, rows, device)
    change_loss = change_criterion(change_logits, changes)

    positive_rows = [index for index, row in enumerate(rows) if row.get("target_label_ids")]
    if not positive_rows:
        return change_loss, change_loss.detach(), change_loss.new_tensor(0.0), change_logits, None

    instruction_emb = encode_all(torch, instruction_encoder, instruction_rows, instruction_batch_size, device)
    positive_index = torch.tensor(positive_rows, dtype=torch.long, device=device)
    logits = matching_logits(torch, F, region_emb.index_select(0, positive_index), instruction_emb, temperature)
    labels = label_matrix(torch, [rows[index] for index in positive_rows], len(instruction_rows), device)
    match_loss = match_criterion(logits, labels)
    return change_loss + match_loss_weight * match_loss, change_loss.detach(), match_loss.detach(), change_logits, logits


def train_epoch(
    torch,
    F,
    region_encoder,
    instruction_encoder,
    change_head,
    optimizer,
    rows: list[dict[str, Any]],
    instruction_rows: list[dict[str, Any]],
    batch_size: int,
    instruction_batch_size: int,
    change_criterion,
    match_criterion,
    temperature: float,
    match_loss_weight: float,
    device: str,
) -> dict[str, float]:
    region_encoder.train()
    instruction_encoder.train()
    change_head.train()
    random.shuffle(rows)
    total = 0
    loss_sum = 0.0
    change_sum = 0.0
    match_sum = 0.0
    for start in range(0, len(rows), batch_size):
        selected = rows[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss, change_loss, match_loss, _change_logits, _match_logits = batch_loss(
            torch,
            F,
            region_encoder,
            instruction_encoder,
            change_head,
            selected,
            instruction_rows,
            instruction_batch_size,
            change_criterion,
            match_criterion,
            temperature,
            match_loss_weight,
            device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(region_encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(instruction_encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(change_head.parameters(), 1.0)
        optimizer.step()
        count = len(selected)
        total += count
        loss_sum += float(loss.item()) * count
        change_sum += float(change_loss.item()) * count
        match_sum += float(match_loss.item()) * count
    return {
        "loss": loss_sum / max(1, total),
        "change_loss": change_sum / max(1, total),
        "match_loss": match_sum / max(1, total),
    }


def evaluate(
    torch,
    F,
    region_encoder,
    instruction_encoder,
    change_head,
    rows: list[dict[str, Any]],
    instruction_rows: list[dict[str, Any]],
    batch_size: int,
    instruction_batch_size: int,
    ks: list[int],
    temperature: float,
    device: str,
) -> dict[str, Any]:
    if not rows:
        return {"examples": 0}
    region_encoder.eval()
    instruction_encoder.eval()
    change_head.eval()
    totals = {k: 0.0 for k in ks}
    positive_examples = 0
    correct_change = 0
    predicted_positive = 0
    true_positive = 0
    actual_positive = 0
    total = 0
    with torch.no_grad():
        instruction_emb = encode_all(torch, instruction_encoder, instruction_rows, instruction_batch_size, device)
        for start in range(0, len(rows), batch_size):
            selected = rows[start : start + batch_size]
            batch = batch_graphs(torch, selected, 0, len(selected), device)
            region_emb = region_encoder(batch)
            change_logits = change_head(region_emb).squeeze(-1)
            changes = change_labels(torch, selected, device)
            predictions = (change_logits > 0).float()
            correct_change += int((predictions == changes).sum().item())
            predicted_positive += int(predictions.sum().item())
            actual_positive += int(changes.sum().item())
            true_positive += int(((predictions == 1) & (changes == 1)).sum().item())
            total += len(selected)

            positive_rows = [index for index, row in enumerate(selected) if row.get("target_label_ids")]
            if not positive_rows:
                continue
            positive_index = torch.tensor(positive_rows, dtype=torch.long, device=device)
            logits = matching_logits(torch, F, region_emb.index_select(0, positive_index), instruction_emb, temperature)
            max_k = min(max(ks), logits.shape[1])
            ranked = torch.topk(logits, k=max_k, dim=1).indices.cpu().tolist()
            for row_offset, rank in zip(positive_rows, ranked):
                gold = set(int(item) for item in selected[row_offset].get("target_label_ids", []))
                if not gold:
                    continue
                for k in ks:
                    totals[k] += len(gold & set(rank[:k])) / len(gold)
                positive_examples += 1
    precision = true_positive / max(1, predicted_positive)
    recall = true_positive / max(1, actual_positive)
    return {
        "examples": total,
        "positive_match_examples": positive_examples,
        "change_accuracy": correct_change / max(1, total),
        "change_precision": precision,
        "change_recall": recall,
        **{f"recall@{k}": totals[k] / max(1, positive_examples) for k in ks},
    }


DEFAULT_EXCLUDE_RECOMMENDATION_LABEL_REGEXES = (
    r"^(?:ir|llvm)\.vector\.(?:load|store)\.",
)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def compile_label_regexes(patterns: list[str] | tuple[str, ...]) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns if pattern]


def keep_recommendation_label(row: dict[str, Any], exclude_regexes: list[re.Pattern[str]]) -> bool:
    label = str(row.get("label", ""))
    return not any(pattern.search(label) for pattern in exclude_regexes)


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
            "probe instruction graph vocab does not match training checkpoint vocab and "
            "some graph rows lack symbolic fields needed for re-encoding "
            f"({joined}). Rebuild probe instruction graphs without --drop-symbols: {instruction_graph_dir}"
        )
    return output


def make_probe_region_record(
    source_file: Path,
    function_name: str,
    region_index: int,
    symbolic: dict[str, Any],
    token_to_id: dict[str, int],
) -> dict[str, Any]:
    encoded = encode_graph(symbolic, token_to_id)
    return {
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
        "node_tokens": symbolic["node_tokens"],
        "node_types": symbolic["node_types"],
        "node_features": symbolic["node_features"],
        "symbolic_edges": symbolic["edges"],
        "edge_features": symbolic["edge_features"],
    }


def build_probe_regions_for_ir(
    ir_file: Path,
    token_to_id: dict[str, int],
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
        except Exception as exc:  # noqa: BLE001
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
            rows.append(
                make_probe_region_record(
                    ir_file,
                    function_name,
                    region_index,
                    subgraph_symbolic(graph, selected),
                    token_to_id,
                )
            )
    return rows, failures


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


def predict_probe_recommendations(
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


def write_probe_markdown(path: Path, rows: list[dict[str, Any]], max_rows: int, top_k: int) -> None:
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
        lines.append(f"- change_probability: `{float(row.get('change_probability', 0.0)):.4f}`")
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


def instruction_label_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key, ""))
        for key in ("label", "instruction", "intrinsic", "extension", "architecture")
        if row.get(key)
    )


def count_target_instruction_rows(rows: list[dict[str, Any]], pattern: re.Pattern[str]) -> int:
    return sum(1 for row in rows if pattern.search(instruction_label_text(row)))


def recommendation_matches_target(item: dict[str, Any], pattern: re.Pattern[str]) -> bool:
    return bool(pattern.search(instruction_label_text(item)))


def summarize_probe_recommendations(rows: list[dict[str, Any]], pattern: re.Pattern[str]) -> dict[str, Any]:
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


def probe_candidate_stem(summary: dict[str, Any]) -> str:
    hit = summary.get("best_target_hit") or {}
    epoch = int(summary.get("epoch", 0))
    rank = int(hit.get("rank", 0))
    hit_regions = int(summary.get("target_hit_regions", 0))
    return f"epoch_{epoch:04d}_rank_{rank:02d}_hits_{hit_regions:03d}"


def unique_artifact_paths(directory: Path, stem: str) -> tuple[Path, Path]:
    checkpoint_path = directory / f"{stem}.pt"
    metadata_path = directory / f"{stem}.json"
    suffix = 2
    while checkpoint_path.exists() or metadata_path.exists():
        checkpoint_path = directory / f"{stem}_{suffix}.pt"
        metadata_path = directory / f"{stem}_{suffix}.json"
        suffix += 1
    return checkpoint_path, metadata_path


def save_probe_candidate_checkpoint(
    torch,
    payload: dict[str, Any],
    summary: dict[str, Any],
    candidate_dir: Path,
) -> dict[str, Any]:
    if not summary.get("best_target_hit"):
        summary["amx_probe_candidate_saved"] = False
        return summary
    candidate_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, metadata_path = unique_artifact_paths(candidate_dir, probe_candidate_stem(summary))
    torch.save(payload, checkpoint_path)
    summary["amx_probe_candidate_saved"] = True
    summary["amx_probe_candidate_checkpoint"] = str(checkpoint_path)
    summary["amx_probe_candidate_metadata"] = str(metadata_path)
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def run_probe_epoch(
    torch,
    F,
    region_encoder,
    instruction_encoder,
    change_head,
    args: argparse.Namespace,
    region_vocab: dict[str, Any],
    instruction_vocab: dict[str, Any],
    device: str,
    epoch: int,
) -> dict[str, Any]:
    graph_instruction_vocab = load_instruction_vocab(args.probe_instruction_graph_dir)
    probe_instruction_rows = reencode_instruction_rows(
        load_instruction_graphs(args.probe_instruction_graph_dir),
        instruction_vocab,
        graph_instruction_vocab,
        args.probe_instruction_graph_dir,
    )
    exclude_label_regexes = compile_label_regexes(DEFAULT_EXCLUDE_RECOMMENDATION_LABEL_REGEXES)
    ranking_instruction_rows = [
        row for row in probe_instruction_rows if keep_recommendation_label(row, exclude_label_regexes)
    ]
    if not ranking_instruction_rows:
        raise SystemExit("no probe instruction graphs remain after recommendation label filtering")

    candidate_modes = parse_candidate_modes(args.probe_candidate_modes)
    probe_regions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for ir_file in args.probe_ir_file:
        file_regions, file_failures = build_probe_regions_for_ir(
            ir_file,
            region_vocab["token_to_id"],
            max_regions_per_function=args.probe_max_regions_per_function,
            candidate_modes=candidate_modes,
            store_slice_depth=args.probe_store_slice_depth,
            store_slice_max_nodes=args.probe_store_slice_max_nodes,
        )
        probe_regions.extend(file_regions)
        failures.extend(file_failures)

    recommendations = predict_probe_recommendations(
        torch,
        F,
        region_encoder,
        instruction_encoder,
        change_head,
        probe_regions,
        ranking_instruction_rows,
        args.batch_size,
        args.instruction_batch_size,
        args.probe_top_k,
        args.temperature,
        device,
    )
    args.probe_out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.probe_out_dir / f"epoch_{epoch:04d}.jsonl"
    out_md = args.probe_out_dir / f"epoch_{epoch:04d}.md"
    write_jsonl(out_jsonl, recommendations)
    write_probe_markdown(out_md, recommendations, args.probe_max_report_regions, args.probe_top_k)

    pattern = re.compile(args.probe_target_regex, re.IGNORECASE)
    summary = summarize_probe_recommendations(recommendations, pattern)
    summary.update(
        {
            "epoch": epoch,
            "out_jsonl": str(out_jsonl),
            "out_md": str(out_md),
            "probe_instruction_graph_dir": str(args.probe_instruction_graph_dir),
            "probe_ir_file": [str(path) for path in args.probe_ir_file],
            "probe_failures": failures[:100],
            "probe_failure_count": len(failures),
        }
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a diff-supervised changed-region detector and intrinsic matcher."
    )
    parser.add_argument("--region-dir", type=Path, required=True)
    parser.add_argument("--instruction-graph-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/llvm_vectorize_regions/models/region_intrinsic_v1"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--instruction-batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--max-edge-position", type=int, default=32)
    parser.add_argument("--max-edge-count", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--match-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--eval-ks", default="1,5,10")
    parser.add_argument("--selection-split", default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--resume-extra-epochs",
        type=int,
        default=0,
        help="When resuming, train this many additional epochs instead of treating --epochs as the final epoch.",
    )
    parser.add_argument(
        "--probe-every-epochs",
        type=int,
        default=0,
        help="Run a zero-shot recommendation probe every N epochs; 0 disables probing.",
    )
    parser.add_argument("--probe-ir-file", type=Path, action="append")
    parser.add_argument("--probe-instruction-graph-dir", type=Path)
    parser.add_argument("--probe-out-dir", type=Path)
    parser.add_argument("--probe-target-regex", default=r"amx|tdpb|tile")
    parser.add_argument("--probe-top-k", type=int, default=20)
    parser.add_argument("--probe-candidate-modes", default="store-backward")
    parser.add_argument("--probe-store-slice-depth", type=int, default=8)
    parser.add_argument("--probe-store-slice-max-nodes", type=int, default=256)
    parser.add_argument("--probe-max-regions-per-function", type=int, default=0)
    parser.add_argument("--probe-max-report-regions", type=int, default=50)
    parser.add_argument("--allow-probe-target-in-train-graphs", action="store_true")
    args = parser.parse_args()

    torch, nn, F = require_torch()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if args.temperature <= 0.0:
        parser.error("--temperature must be positive")
    if args.match_loss_weight < 0.0:
        parser.error("--match-loss-weight must be non-negative")
    if args.resume_extra_epochs < 0:
        parser.error("--resume-extra-epochs must be non-negative")
    if args.probe_every_epochs < 0:
        parser.error("--probe-every-epochs must be non-negative")
    if args.probe_every_epochs > 0:
        if not args.probe_ir_file:
            parser.error("--probe-ir-file is required when --probe-every-epochs is positive")
        if args.probe_instruction_graph_dir is None:
            parser.error("--probe-instruction-graph-dir is required when --probe-every-epochs is positive")
        if args.probe_top_k <= 0:
            parser.error("--probe-top-k must be positive")
        try:
            parse_candidate_modes(args.probe_candidate_modes)
        except ValueError as exc:
            parser.error(str(exc))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    region_vocab = json.loads((args.region_dir / "vocab.json").read_text(encoding="utf-8"))
    instruction_vocab = json.loads((args.instruction_graph_dir / "vocab.json").read_text(encoding="utf-8"))
    instruction_rows_all = load_instruction_graphs(args.instruction_graph_dir)
    train_instruction_rows = [row for row in instruction_rows_all if str(row.get("split", "train")) == "train"]
    if not train_instruction_rows:
        train_instruction_rows = instruction_rows_all

    probe_enabled = args.probe_every_epochs > 0
    if probe_enabled:
        if args.probe_out_dir is None:
            args.probe_out_dir = args.out_dir / "amx_probe"
        args.probe_out_dir.mkdir(parents=True, exist_ok=True)
        target_pattern = re.compile(args.probe_target_regex, re.IGNORECASE)
        train_target_count = count_target_instruction_rows(instruction_rows_all, target_pattern)
        probe_instruction_rows_for_check = load_instruction_graphs(args.probe_instruction_graph_dir)
        probe_target_count = count_target_instruction_rows(probe_instruction_rows_for_check, target_pattern)
        if train_target_count and not args.allow_probe_target_in_train_graphs:
            raise SystemExit(
                "training instruction graph directory appears to contain probe target labels "
                f"({train_target_count} matches). Use a no-AMX training graph directory, or pass "
                "--allow-probe-target-in-train-graphs intentionally."
            )
        if not probe_target_count:
            raise SystemExit(
                "probe instruction graph directory has no probe target labels. "
                "Build it with AMX specs first."
            )
        print(
            json.dumps(
                {
                    "probe_target_regex": args.probe_target_regex,
                    "train_probe_target_label_matches": train_target_count,
                    "probe_target_label_matches": probe_target_count,
                    "probe_every_epochs": args.probe_every_epochs,
                },
                sort_keys=True,
            )
        )

    train_path = region_path(args.region_dir, "train")
    if not train_path.exists():
        raise SystemExit(f"missing train split: {train_path}")
    train_rows = align_region_rows(load_jsonl(train_path), train_instruction_rows)
    if not train_rows:
        raise SystemExit("empty train region split")
    if not any(row.get("target_label_ids") for row in train_rows):
        raise SystemExit("no train changed-region rows align with instruction graph labels")

    eval_rows: dict[str, list[dict[str, Any]]] = {"train": train_rows}
    eval_instruction_rows: dict[str, list[dict[str, Any]]] = {"train": train_instruction_rows}
    for split in ("val", "test"):
        path = region_path(args.region_dir, split)
        if path.exists():
            eval_rows[split] = align_region_rows(load_jsonl(path), instruction_rows_all)
            eval_instruction_rows[split] = instruction_rows_all

    region_encoder = RelationalGraphEncoder(
        torch,
        nn,
        F,
        token_vocab_size=len(region_vocab["token_to_id"]),
        node_type_count=len(region_vocab["node_type_to_id"]),
        edge_type_count=len(region_vocab["edge_type_to_id"]),
        edge_feature_count=len(region_vocab.get("edge_feature_to_id", {"none": 0})),
        max_edge_position=args.max_edge_position,
        max_edge_count=args.max_edge_count,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
    ).model.to(device)
    instruction_encoder = RelationalGraphEncoder(
        torch,
        nn,
        F,
        token_vocab_size=len(instruction_vocab["token_to_id"]),
        node_type_count=len(instruction_vocab["node_type_to_id"]),
        edge_type_count=len(instruction_vocab["edge_type_to_id"]),
        edge_feature_count=len(instruction_vocab.get("edge_feature_to_id", {"none": 0})),
        max_edge_position=args.max_edge_position,
        max_edge_count=args.max_edge_count,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
    ).model.to(device)
    change_head = nn.Linear(args.hidden_dim, 1).to(device)
    optimizer = torch.optim.AdamW(
        list(region_encoder.parameters()) + list(instruction_encoder.parameters()) + list(change_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    resume_epoch = 0
    best_score = -math.inf
    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location=device)
        region_encoder.load_state_dict(checkpoint["region_encoder_state"])
        instruction_encoder.load_state_dict(checkpoint["instruction_encoder_state"])
        change_head.load_state_dict(checkpoint["change_head_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        resume_epoch = int(checkpoint.get("epoch", 0))
        best_score = float(checkpoint.get("best_score", checkpoint.get("score", -math.inf)))
        print(
            json.dumps(
                {
                    "resume_from": str(args.resume_from),
                    "resume_epoch": resume_epoch,
                    "loaded_optimizer_state": "optimizer_state" in checkpoint,
                    "best_score": best_score,
                },
                sort_keys=True,
            )
        )
    change_criterion = nn.BCEWithLogitsLoss(
        pos_weight=change_positive_weight(torch, train_rows, args.max_pos_weight, device)
    )
    match_criterion = nn.BCEWithLogitsLoss(
        pos_weight=match_positive_weight(torch, train_rows, len(train_instruction_rows), args.max_pos_weight, device)
    )
    ks = [int(item) for item in args.eval_ks.split(",") if item.strip()]
    final_epoch = args.epochs
    if args.resume_from is not None and args.resume_extra_epochs > 0:
        final_epoch = resume_epoch + args.resume_extra_epochs
    start_epoch = resume_epoch + 1
    if final_epoch < start_epoch:
        raise SystemExit(
            f"nothing to train: checkpoint epoch is {resume_epoch}, final epoch is {final_epoch}; "
            "increase --epochs or pass --resume-extra-epochs"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "region_dir": str(args.region_dir),
        "instruction_graph_dir": str(args.instruction_graph_dir),
        "device": device,
        "epochs": final_epoch,
        "batch_size": args.batch_size,
        "instruction_batch_size": args.instruction_batch_size,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "max_edge_position": args.max_edge_position,
        "max_edge_count": args.max_edge_count,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "match_loss_weight": args.match_loss_weight,
        "max_pos_weight": args.max_pos_weight,
        "eval_ks": ks,
        "selection_split": args.selection_split,
        "seed": args.seed,
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "resume_epoch": resume_epoch,
        "resume_extra_epochs": args.resume_extra_epochs,
        "num_train_instruction_graphs": len(train_instruction_rows),
        "num_eval_instruction_graphs": len(instruction_rows_all),
        "probe_every_epochs": args.probe_every_epochs,
        "probe_ir_file": [str(path) for path in args.probe_ir_file or []],
        "probe_instruction_graph_dir": str(args.probe_instruction_graph_dir) if args.probe_instruction_graph_dir else None,
        "probe_out_dir": str(args.probe_out_dir) if args.probe_out_dir else None,
        "probe_target_regex": args.probe_target_regex,
        "probe_top_k": args.probe_top_k,
        "probe_candidate_modes": args.probe_candidate_modes,
        "probe_store_slice_depth": args.probe_store_slice_depth,
        "probe_store_slice_max_nodes": args.probe_store_slice_max_nodes,
        "probe_candidate_checkpoint_dir": str(args.out_dir / "amx_probe_candidates") if probe_enabled else None,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_path = args.out_dir / "best.pt"
    last_path = args.out_dir / "last.pt"
    probe_candidate_dir = args.out_dir / "amx_probe_candidates"
    history_path = args.out_dir / "history.json"
    probe_history_path = (args.probe_out_dir / "probe_history.jsonl") if probe_enabled else None
    history: list[dict[str, Any]] = []
    probe_history: list[dict[str, Any]] = []
    if args.resume_from is not None and history_path.exists():
        try:
            previous_history = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(previous_history, list):
                history = [row for row in previous_history if int(row.get("epoch", 0)) < start_epoch]
        except json.JSONDecodeError:
            history = []
    if probe_enabled and probe_history_path and probe_history_path.exists():
        probe_history = load_jsonl(probe_history_path)

    def checkpoint_payload(epoch: int, score: float) -> dict[str, Any]:
        return {
            "region_encoder_state": region_encoder.state_dict(),
            "instruction_encoder_state": instruction_encoder.state_dict(),
            "change_head_state": change_head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "region_vocab": region_vocab,
            "instruction_vocab": instruction_vocab,
            "epoch": epoch,
            "score": score,
            "best_score": best_score,
        }

    for epoch in range(start_epoch, final_epoch + 1):
        train_metrics = train_epoch(
            torch,
            F,
            region_encoder,
            instruction_encoder,
            change_head,
            optimizer,
            train_rows,
            train_instruction_rows,
            args.batch_size,
            args.instruction_batch_size,
            change_criterion,
            match_criterion,
            args.temperature,
            args.match_loss_weight,
            device,
        )
        metrics = {
            split: evaluate(
                torch,
                F,
                region_encoder,
                instruction_encoder,
                change_head,
                rows,
                eval_instruction_rows[split],
                args.batch_size,
                args.instruction_batch_size,
                ks,
                args.temperature,
                device,
            )
            for split, rows in eval_rows.items()
        }
        selected = metrics.get(args.selection_split, metrics.get("train", {}))
        score = float(selected.get("recall@5", selected.get("change_accuracy", 0.0)))
        row = {"epoch": epoch, "train": train_metrics, **metrics}
        is_best = score > best_score
        if is_best:
            best_score = score
        payload = checkpoint_payload(epoch, score)
        torch.save(payload, last_path)
        if is_best:
            torch.save(payload, best_path)
        if probe_enabled and epoch % args.probe_every_epochs == 0:
            probe_summary = run_probe_epoch(
                torch,
                F,
                region_encoder,
                instruction_encoder,
                change_head,
                args,
                region_vocab,
                instruction_vocab,
                device,
                epoch,
            )
            probe_summary = save_probe_candidate_checkpoint(
                torch,
                payload,
                probe_summary,
                probe_candidate_dir,
            )
            probe_history.append(probe_summary)
            if probe_history_path:
                write_jsonl(probe_history_path, probe_history)
            row["probe"] = probe_summary
        history.append(row)
        print(json.dumps(row, sort_keys=True))

    final_metrics = {
        split: evaluate(
            torch,
            F,
            region_encoder,
            instruction_encoder,
            change_head,
            rows,
            eval_instruction_rows[split],
            args.batch_size,
            args.instruction_batch_size,
            ks,
            args.temperature,
            device,
        )
        for split, rows in eval_rows.items()
    }
    final_metrics["best_score"] = best_score
    final_metrics["best_checkpoint"] = str(best_path)
    final_metrics["last_checkpoint"] = str(last_path)
    if probe_enabled:
        final_metrics["amx_probe_candidate_checkpoint_dir"] = str(probe_candidate_dir)
        final_metrics["amx_probe_candidate_checkpoints"] = [
            str(row["amx_probe_candidate_checkpoint"])
            for row in probe_history
            if row.get("amx_probe_candidate_checkpoint")
        ]
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.out_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "metrics": final_metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
