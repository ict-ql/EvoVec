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


def instruction_graph_paths(graph_dir: Path) -> list[Path]:
    return sorted(graph_dir.glob("*.instruction_graphs.jsonl"))


def set_seed(seed: int) -> None:
    random.seed(seed)


def batch_graphs(torch, rows: list[dict[str, Any]], start: int, batch_size: int, device: str) -> dict[str, Any]:
    selected = rows[start : start + batch_size]
    node_tokens: list[int] = []
    node_types: list[int] = []
    graph_ids: list[int] = []
    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_types: list[int] = []
    edge_positions: list[int] = []
    edge_features: list[int] = []
    edge_counts: list[int] = []

    node_offset = 0
    for graph_index, row in enumerate(selected):
        tokens = [int(item) for item in row["node_token_ids"]]
        types = [int(item) for item in row["node_type_ids"]]
        node_tokens.extend(tokens)
        node_types.extend(types)
        graph_ids.extend([graph_index] * len(tokens))
        row_edge_features = row.get("edge_feature_ids", [])
        row_edge_counts = row.get("edge_counts", [])
        for edge_index, edge in enumerate(row["edges"]):
            src, dst, edge_type = edge[:3]
            edge_src.append(int(src) + node_offset)
            edge_dst.append(int(dst) + node_offset)
            edge_types.append(int(edge_type))
            edge_positions.append(int(edge[3]) if len(edge) > 3 else 0)
            edge_features.append(int(row_edge_features[edge_index]) if edge_index < len(row_edge_features) else 0)
            edge_counts.append(int(row_edge_counts[edge_index]) if edge_index < len(row_edge_counts) else 0)
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
        "edge_positions": torch.tensor(edge_positions, dtype=torch.long, device=device),
        "edge_features": torch.tensor(edge_features, dtype=torch.long, device=device),
        "edge_counts": torch.tensor(edge_counts, dtype=torch.long, device=device),
        "batch_size": len(selected),
    }


def label_matrix(torch, rows: list[dict[str, Any]], start: int, batch_size: int, num_labels: int, device: str):
    selected = rows[start : start + batch_size]
    labels = torch.zeros((len(selected), num_labels), dtype=torch.float32, device=device)
    for row_index, row in enumerate(selected):
        for label_id in row["candidate_label_ids"]:
            if 0 <= int(label_id) < num_labels:
                labels[row_index, int(label_id)] = 1.0
    return labels


class RelationalGraphEncoder:
    def __init__(
        self,
        torch,
        nn,
        F,
        token_vocab_size: int,
        node_type_count: int,
        edge_type_count: int,
        edge_feature_count: int,
        max_edge_position: int,
        max_edge_count: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.token_emb = nn.Embedding(token_vocab_size, hidden_dim, padding_idx=0)
                self.type_emb = nn.Embedding(node_type_count, hidden_dim)
                self.position_emb = nn.Embedding(max_edge_position + 1, hidden_dim)
                self.edge_feature_emb = nn.Embedding(edge_feature_count, hidden_dim)
                self.edge_count_emb = nn.Embedding(max_edge_count + 1, hidden_dim)
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
                self.proj = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )

            def forward(self, batch: dict[str, Any]):
                h = self.input_norm(self.token_emb(batch["node_tokens"]) + self.type_emb(batch["node_types"]))
                edge_src = batch["edge_src"]
                edge_dst = batch["edge_dst"]
                edge_types = batch["edge_types"]
                edge_positions = batch["edge_positions"].clamp(max=self.position_emb.num_embeddings - 1)
                edge_features = batch["edge_features"].clamp(max=self.edge_feature_emb.num_embeddings - 1)
                edge_counts = batch["edge_counts"].clamp(max=self.edge_count_emb.num_embeddings - 1)
                for layer_index, self_linear in enumerate(self.self_linears):
                    agg = torch.zeros_like(h)
                    if edge_src.numel() > 0:
                        for edge_type, edge_linear in enumerate(self.edge_linears[layer_index]):
                            mask = edge_types == edge_type
                            if mask.any():
                                positioned = (
                                    h[edge_src[mask]]
                                    + self.position_emb(edge_positions[mask])
                                    + self.edge_feature_emb(edge_features[mask])
                                    + self.edge_count_emb(edge_counts[mask])
                                )
                                msg = edge_linear(positioned)
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
                pooled = torch.cat([mean_pool, torch.stack(max_pool, dim=0)], dim=1)
                return self.proj(pooled)

        self.model = Model()


def load_instruction_graphs(instruction_graph_dir: Path) -> list[dict[str, Any]]:
    paths = instruction_graph_paths(instruction_graph_dir)
    if not paths:
        raise SystemExit(f"no *.instruction_graphs.jsonl files found in {instruction_graph_dir}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_jsonl(path))
    return rows


def align_program_rows(
    rows: list[dict[str, Any]],
    program_vocab: dict[str, Any],
    instruction_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    id_to_instruction = {
        int(label_id): instruction for instruction, label_id in program_vocab["instruction_to_id"].items()
    }
    instruction_by_label: dict[str, dict[str, Any]] = {}
    for row in instruction_rows:
        keys = [str(row["label"])]
        keys.extend(str(item) for item in row.get("aliases", []) if str(item))
        for key in keys:
            instruction_by_label.setdefault(key, row)
    candidate_labels: list[str] = []
    candidates: list[dict[str, Any]] = []
    label_to_candidate: dict[str, int] = {}
    canonical_to_candidate: dict[str, int] = {}
    for _label_id, label in sorted(id_to_instruction.items()):
        row = instruction_by_label.get(label)
        if row is None:
            continue
        canonical = str(row["label"])
        if canonical not in canonical_to_candidate:
            canonical_to_candidate[canonical] = len(candidates)
            candidate_labels.append(canonical)
            candidates.append(row)
        label_to_candidate[label] = canonical_to_candidate[canonical]
    if not candidates:
        raise SystemExit("none of the program labels have matching instruction semantic graphs")

    aligned: list[dict[str, Any]] = []
    for row in rows:
        candidate_label_ids = sorted(
            {
                label_to_candidate[id_to_instruction[int(label_id)]]
                for label_id in row["label_ids"]
                if int(label_id) in id_to_instruction
                and id_to_instruction[int(label_id)] in label_to_candidate
            }
        )
        if not candidate_label_ids:
            continue
        aligned_row = dict(row)
        aligned_row["candidate_label_ids"] = candidate_label_ids
        aligned.append(aligned_row)
    return aligned, candidate_labels, candidates


def encode_all(torch, encoder, rows: list[dict[str, Any]], batch_size: int, device: str):
    chunks = []
    for start in range(0, len(rows), batch_size):
        batch = batch_graphs(torch, rows, start, batch_size, device)
        chunks.append(encoder(batch))
    return torch.cat(chunks, dim=0)


def positive_weight(torch, rows: list[dict[str, Any]], num_labels: int, max_weight: float, device: str):
    positives = torch.zeros((num_labels,), dtype=torch.float32)
    for row in rows:
        for label_id in row["candidate_label_ids"]:
            positives[int(label_id)] += 1.0
    total = max(1, len(rows))
    negatives = total - positives
    weights = negatives / positives.clamp_min(1.0)
    return torch.clamp(weights, min=1.0, max=max_weight).to(device)


def evaluate(
    torch,
    F,
    program_encoder,
    instruction_encoder,
    rows: list[dict[str, Any]],
    instruction_rows: list[dict[str, Any]],
    batch_size: int,
    instruction_batch_size: int,
    ks: list[int],
    temperature: float,
    device: str,
) -> dict[str, Any]:
    if not rows:
        return {"examples": 0, **{f"recall@{k}": 0.0 for k in ks}}
    program_encoder.eval()
    instruction_encoder.eval()
    num_labels = len(instruction_rows)
    totals = {k: 0.0 for k in ks}
    examples = 0
    loss_sum = 0.0
    with torch.no_grad():
        instruction_emb = F.normalize(
            encode_all(torch, instruction_encoder, instruction_rows, instruction_batch_size, device), dim=1
        )
        for start in range(0, len(rows), batch_size):
            batch = batch_graphs(torch, rows, start, batch_size, device)
            program_emb = F.normalize(program_encoder(batch), dim=1)
            logits = program_emb @ instruction_emb.T / temperature
            labels = label_matrix(torch, rows, start, batch_size, num_labels, device)
            loss_sum += float(F.binary_cross_entropy_with_logits(logits, labels).item()) * labels.shape[0]
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
    return {
        "examples": examples,
        "loss": loss_sum / max(1, len(rows)),
        **{f"recall@{k}": totals[k] / max(1, examples) for k in ks},
    }


def train_epoch(
    torch,
    F,
    program_encoder,
    instruction_encoder,
    optimizer,
    criterion,
    rows: list[dict[str, Any]],
    instruction_rows: list[dict[str, Any]],
    batch_size: int,
    instruction_batch_size: int,
    temperature: float,
    device: str,
) -> float:
    program_encoder.train()
    instruction_encoder.train()
    random.shuffle(rows)
    total_loss = 0.0
    total_examples = 0
    for start in range(0, len(rows), batch_size):
        optimizer.zero_grad(set_to_none=True)
        instruction_emb = F.normalize(
            encode_all(torch, instruction_encoder, instruction_rows, instruction_batch_size, device), dim=1
        )
        batch = batch_graphs(torch, rows, start, batch_size, device)
        program_emb = F.normalize(program_encoder(batch), dim=1)
        logits = program_emb @ instruction_emb.T / temperature
        labels = label_matrix(torch, rows, start, batch_size, len(instruction_rows), device)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(program_encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(instruction_encoder.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item()) * labels.shape[0]
        total_examples += labels.shape[0]
    return total_loss / max(1, total_examples)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a dual-graph GNN retriever over program graphs and instruction Operation graphs."
    )
    parser.add_argument("--program-graph-dir", type=Path, required=True)
    parser.add_argument("--instruction-graph-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/iroptset/models/dual_gnn_pilot_10k_v1"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--instruction-batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--max-edge-position", type=int, default=32)
    parser.add_argument("--max-edge-count", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--eval-ks", default="10,20,50")
    parser.add_argument("--selection-split", default="train")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    torch, nn, F = require_torch()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if args.temperature <= 0.0:
        parser.error("--temperature must be positive")
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    program_vocab = json.loads((args.program_graph_dir / "vocab.json").read_text(encoding="utf-8"))
    instruction_vocab = json.loads((args.instruction_graph_dir / "vocab.json").read_text(encoding="utf-8"))
    instruction_rows_all = load_instruction_graphs(args.instruction_graph_dir)

    train_raw = load_jsonl(graph_path(args.program_graph_dir, "train"))
    if not train_raw:
        raise SystemExit("empty train graph split")
    train_instruction_rows = [row for row in instruction_rows_all if str(row.get("split", "train")) == "train"]
    if not train_instruction_rows:
        raise SystemExit("instruction graph directory has no split=train rows")
    train_rows, train_candidate_labels, train_instruction_rows = align_program_rows(
        train_raw, program_vocab, train_instruction_rows
    )
    if not train_rows:
        raise SystemExit("no train rows remain after aligning labels with instruction graphs")

    eval_rows: dict[str, list[dict[str, Any]]] = {}
    eval_instruction_rows: dict[str, list[dict[str, Any]]] = {}
    eval_rows["train"] = train_rows
    eval_instruction_rows["train"] = train_instruction_rows
    for split in ("val", "test"):
        path = graph_path(args.program_graph_dir, split)
        if path.exists():
            rows, _labels, _candidates = align_program_rows(load_jsonl(path), program_vocab, instruction_rows_all)
            eval_rows[split] = rows
            eval_instruction_rows[split] = instruction_rows_all

    program_encoder = RelationalGraphEncoder(
        torch,
        nn,
        F,
        token_vocab_size=len(program_vocab["token_to_id"]),
        node_type_count=len(program_vocab["node_type_to_id"]),
        edge_type_count=len(program_vocab["edge_type_to_id"]),
        edge_feature_count=len(program_vocab.get("edge_feature_to_id", {"none": 0})),
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
    optimizer = torch.optim.AdamW(
        list(program_encoder.parameters()) + list(instruction_encoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight(torch, train_rows, len(train_instruction_rows), args.max_pos_weight, device)
    )
    ks = [int(item) for item in args.eval_ks.split(",") if item.strip()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "program_graph_dir": str(args.program_graph_dir),
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
        "max_pos_weight": args.max_pos_weight,
        "eval_ks": ks,
        "selection_split": args.selection_split,
        "seed": args.seed,
        "num_train_candidate_labels": len(train_candidate_labels),
        "num_eval_candidate_labels": len(instruction_rows_all),
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.out_dir / "train_candidate_labels.json").write_text(
        json.dumps(train_candidate_labels, indent=2), encoding="utf-8"
    )
    (args.out_dir / "eval_candidate_labels.json").write_text(
        json.dumps([str(row["label"]) for row in instruction_rows_all], indent=2), encoding="utf-8"
    )

    best_score = -math.inf
    history: list[dict[str, Any]] = []
    best_path = args.out_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            torch,
            F,
            program_encoder,
            instruction_encoder,
            optimizer,
            criterion,
            train_rows,
            train_instruction_rows,
            args.batch_size,
            args.instruction_batch_size,
            args.temperature,
            device,
        )
        metrics = {
            split: evaluate(
                torch,
                F,
                program_encoder,
                instruction_encoder,
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
        score = metrics.get(args.selection_split, metrics["train"]).get("recall@20", 0.0)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if score > best_score:
            best_score = float(score)
            torch.save(
                {
                    "program_encoder_state": program_encoder.state_dict(),
                    "instruction_encoder_state": instruction_encoder.state_dict(),
                    "config": config,
                    "program_vocab": program_vocab,
                    "instruction_vocab": instruction_vocab,
                    "train_candidate_labels": train_candidate_labels,
                    "eval_candidate_labels": [str(row["label"]) for row in instruction_rows_all],
                    "epoch": epoch,
                    "score": best_score,
                },
                best_path,
            )

    final_metrics = {
        split: evaluate(
            torch,
            F,
            program_encoder,
            instruction_encoder,
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
