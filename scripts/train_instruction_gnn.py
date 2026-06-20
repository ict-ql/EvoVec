#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def graph_path(graph_dir: Path, split: str) -> Path:
    return graph_dir / f"{split}.graphs.jsonl"


def set_seed(seed: int) -> None:
    random.seed(seed)


def batch_graphs(rows: list[dict[str, Any]], start: int, batch_size: int, num_labels: int, device: str):
    torch, _nn, _F = require_torch()
    selected = rows[start : start + batch_size]
    node_tokens: list[int] = []
    node_types: list[int] = []
    graph_ids: list[int] = []
    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_types: list[int] = []
    labels = torch.zeros((len(selected), num_labels), dtype=torch.float32)

    node_offset = 0
    for graph_index, row in enumerate(selected):
        tokens = [int(item) for item in row["node_token_ids"]]
        types = [int(item) for item in row["node_type_ids"]]
        node_tokens.extend(tokens)
        node_types.extend(types)
        graph_ids.extend([graph_index] * len(tokens))
        for src, dst, edge_type in row["edges"]:
            edge_src.append(int(src) + node_offset)
            edge_dst.append(int(dst) + node_offset)
            edge_types.append(int(edge_type))
        for label_id in row["label_ids"]:
            if 0 <= int(label_id) < num_labels:
                labels[graph_index, int(label_id)] = 1.0
        node_offset += len(tokens)

    if not node_tokens:
        node_tokens = [0]
        node_types = [0]
        graph_ids = [0]
    return {
        "node_tokens": torch.tensor(node_tokens, dtype=torch.long, device=device),
        "node_types": torch.tensor(node_types, dtype=torch.long, device=device),
        "graph_ids": torch.tensor(graph_ids, dtype=torch.long, device=device),
        "edge_src": torch.tensor(edge_src, dtype=torch.long, device=device),
        "edge_dst": torch.tensor(edge_dst, dtype=torch.long, device=device),
        "edge_types": torch.tensor(edge_types, dtype=torch.long, device=device),
        "labels": labels.to(device),
        "batch_size": len(selected),
    }


class RelationalGraphClassifier:
    def __init__(
        self,
        torch,
        nn,
        F,
        token_vocab_size: int,
        node_type_count: int,
        edge_type_count: int,
        num_labels: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.token_emb = nn.Embedding(token_vocab_size, hidden_dim, padding_idx=0)
                self.type_emb = nn.Embedding(node_type_count, hidden_dim)
                self.input_norm = nn.LayerNorm(hidden_dim)
                self.self_linears = nn.ModuleList(
                    [nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)]
                )
                self.edge_linears = nn.ModuleList(
                    [
                        nn.ModuleList(
                            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(edge_type_count)]
                        )
                        for _ in range(layers)
                    ]
                )
                self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
                self.dropout = dropout
                self.out = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_labels),
                )

            def forward(self, batch: dict[str, Any]):
                h = self.input_norm(self.token_emb(batch["node_tokens"]) + self.type_emb(batch["node_types"]))
                edge_src = batch["edge_src"]
                edge_dst = batch["edge_dst"]
                edge_types = batch["edge_types"]
                for layer_index, self_linear in enumerate(self.self_linears):
                    agg = torch.zeros_like(h)
                    if edge_src.numel() > 0:
                        for edge_type, edge_linear in enumerate(self.edge_linears[layer_index]):
                            mask = edge_types == edge_type
                            if mask.any():
                                msg = edge_linear(h[edge_src[mask]])
                                agg.index_add_(0, edge_dst[mask], msg)
                    deg = torch.ones((h.shape[0], 1), dtype=h.dtype, device=h.device)
                    if edge_dst.numel() > 0:
                        ones = torch.ones((edge_dst.shape[0], 1), dtype=h.dtype, device=h.device)
                        deg.index_add_(0, edge_dst, ones)
                    updated = self_linear(h) + agg / deg.clamp_min(1.0)
                    h = self.norms[layer_index](F.relu(updated))
                    h = F.dropout(h, p=self.dropout, training=self.training)

                graph_ids = batch["graph_ids"]
                batch_size = batch["batch_size"]
                mean_pool = torch.zeros((batch_size, h.shape[1]), dtype=h.dtype, device=h.device)
                mean_pool.index_add_(0, graph_ids, h)
                counts = torch.zeros((batch_size, 1), dtype=h.dtype, device=h.device)
                counts.index_add_(0, graph_ids, torch.ones((h.shape[0], 1), dtype=h.dtype, device=h.device))
                mean_pool = mean_pool / counts.clamp_min(1.0)

                max_pool = []
                for graph_index in range(batch_size):
                    nodes = h[graph_ids == graph_index]
                    if nodes.numel() == 0:
                        max_pool.append(torch.zeros((h.shape[1],), dtype=h.dtype, device=h.device))
                    else:
                        max_pool.append(nodes.max(dim=0).values)
                max_pool_tensor = torch.stack(max_pool, dim=0)
                pooled = torch.cat([mean_pool, max_pool_tensor], dim=1)
                return self.out(pooled)

        self.model = Model()


def label_pos_weight(rows: list[dict[str, Any]], num_labels: int, max_weight: float, device: str):
    torch, _nn, _F = require_torch()
    positives = torch.zeros((num_labels,), dtype=torch.float32)
    for row in rows:
        for label_id in row["label_ids"]:
            positives[int(label_id)] += 1.0
    total = max(1, len(rows))
    negatives = total - positives
    weights = negatives / positives.clamp_min(1.0)
    weights = torch.clamp(weights, min=1.0, max=max_weight)
    return weights.to(device)


def evaluate(model, rows: list[dict[str, Any]], batch_size: int, num_labels: int, ks: list[int], device: str) -> dict[str, Any]:
    torch, _nn, _F = require_torch()
    model.eval()
    totals = {k: 0.0 for k in ks}
    examples = 0
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = batch_graphs(rows, start, batch_size, num_labels, device)
            logits = model(batch)
            labels = batch["labels"]
            loss_sum += float(torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()) * labels.shape[0]
            max_k = min(max(ks), num_labels)
            top = torch.topk(logits, k=max_k, dim=1).indices.cpu().tolist()
            gold = labels.cpu()
            for row_index, ranked in enumerate(top):
                gold_ids = set(torch.nonzero(gold[row_index] > 0.5, as_tuple=False).flatten().tolist())
                if not gold_ids:
                    continue
                for k in ks:
                    totals[k] += len(gold_ids & set(ranked[:k])) / len(gold_ids)
                examples += 1
    if examples == 0:
        return {"examples": 0, "loss": 0.0, **{f"recall@{k}": 0.0 for k in ks}}
    return {
        "examples": examples,
        "loss": loss_sum / max(1, len(rows)),
        **{f"recall@{k}": totals[k] / examples for k in ks},
    }


def train_epoch(model, optimizer, criterion, rows: list[dict[str, Any]], batch_size: int, num_labels: int, device: str) -> float:
    torch, _nn, _F = require_torch()
    model.train()
    random.shuffle(rows)
    total_loss = 0.0
    total_examples = 0
    for start in range(0, len(rows), batch_size):
        batch = batch_graphs(rows, start, batch_size, num_labels, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item()) * batch["labels"].shape[0]
        total_examples += batch["labels"].shape[0]
    return total_loss / max(1, total_examples)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a pure-PyTorch GNN for intrinsic instruction retrieval.")
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/iroptset/models/gnn_pilot_10k_v1"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--eval-ks", default="10,20,50")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    torch, nn, F = require_torch()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    vocab = json.loads((args.graph_dir / "vocab.json").read_text(encoding="utf-8"))
    train_rows = load_jsonl(graph_path(args.graph_dir, "train"))
    if not train_rows:
        raise SystemExit("empty train graph split")
    val_rows = load_jsonl(graph_path(args.graph_dir, "val")) if graph_path(args.graph_dir, "val").exists() else []
    test_rows = load_jsonl(graph_path(args.graph_dir, "test")) if graph_path(args.graph_dir, "test").exists() else []

    num_labels = len(vocab["instruction_to_id"])
    token_vocab_size = len(vocab["token_to_id"])
    node_type_count = len(vocab["node_type_to_id"])
    edge_type_count = len(vocab["edge_type_to_id"])
    wrapper = RelationalGraphClassifier(
        torch,
        nn,
        F,
        token_vocab_size,
        node_type_count,
        edge_type_count,
        num_labels,
        args.hidden_dim,
        args.layers,
        args.dropout,
    )
    model = wrapper.model.to(device)
    pos_weight = label_pos_weight(train_rows, num_labels, args.max_pos_weight, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ks = [int(item) for item in args.eval_ks.split(",") if item.strip()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "graph_dir": str(args.graph_dir),
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "layers": args.layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_pos_weight": args.max_pos_weight,
        "eval_ks": ks,
        "seed": args.seed,
        "num_labels": num_labels,
        "token_vocab_size": token_vocab_size,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_score = -math.inf
    history: list[dict[str, Any]] = []
    best_path = args.out_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, optimizer, criterion, train_rows, args.batch_size, num_labels, device)
        train_metrics = evaluate(model, train_rows, args.batch_size, num_labels, ks, device)
        val_metrics = evaluate(model, val_rows, args.batch_size, num_labels, ks, device) if val_rows else {}
        score = val_metrics.get("recall@20", train_metrics.get("recall@20", 0.0))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if score > best_score:
            best_score = float(score)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "vocab": vocab,
                    "epoch": epoch,
                    "score": best_score,
                },
                best_path,
            )

    final_metrics = {
        "train": evaluate(model, train_rows, args.batch_size, num_labels, ks, device),
        "val": evaluate(model, val_rows, args.batch_size, num_labels, ks, device) if val_rows else {},
        "test": evaluate(model, test_rows, args.batch_size, num_labels, ks, device) if test_rows else {},
        "best_score": best_score,
        "best_checkpoint": str(best_path),
    }
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.out_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "metrics": final_metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
