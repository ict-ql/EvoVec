#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any

try:
    from train_instruction_dual_gnn import (
        RelationalGraphEncoder,
        batch_graphs,
        encode_all,
        instruction_graph_paths,
        load_jsonl,
    )
except ModuleNotFoundError:  # pragma: no cover - used when imported as a package module
    from scripts.train_instruction_dual_gnn import (
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
    args = parser.parse_args()

    torch, nn, F = require_torch()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if args.temperature <= 0.0:
        parser.error("--temperature must be positive")
    if args.match_loss_weight < 0.0:
        parser.error("--match-loss-weight must be non-negative")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    region_vocab = json.loads((args.region_dir / "vocab.json").read_text(encoding="utf-8"))
    instruction_vocab = json.loads((args.instruction_graph_dir / "vocab.json").read_text(encoding="utf-8"))
    instruction_rows_all = load_instruction_graphs(args.instruction_graph_dir)
    train_instruction_rows = [row for row in instruction_rows_all if str(row.get("split", "train")) == "train"]
    if not train_instruction_rows:
        train_instruction_rows = instruction_rows_all

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
    change_criterion = nn.BCEWithLogitsLoss(
        pos_weight=change_positive_weight(torch, train_rows, args.max_pos_weight, device)
    )
    match_criterion = nn.BCEWithLogitsLoss(
        pos_weight=match_positive_weight(torch, train_rows, len(train_instruction_rows), args.max_pos_weight, device)
    )
    ks = [int(item) for item in args.eval_ks.split(",") if item.strip()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "region_dir": str(args.region_dir),
        "instruction_graph_dir": str(args.instruction_graph_dir),
        "device": device,
        "epochs": args.epochs,
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
        "num_train_instruction_graphs": len(train_instruction_rows),
        "num_eval_instruction_graphs": len(instruction_rows_all),
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_score = -math.inf
    best_path = args.out_dir / "best.pt"
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
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
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "region_encoder_state": region_encoder.state_dict(),
                    "instruction_encoder_state": instruction_encoder.state_dict(),
                    "change_head_state": change_head.state_dict(),
                    "config": config,
                    "region_vocab": region_vocab,
                    "instruction_vocab": instruction_vocab,
                    "epoch": epoch,
                    "score": best_score,
                },
                best_path,
            )

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
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.out_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "metrics": final_metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
