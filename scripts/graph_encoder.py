from __future__ import annotations

import json
from pathlib import Path
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


def instruction_graph_paths(graph_dir: Path) -> list[Path]:
    return sorted(graph_dir.glob("*.instruction_graphs.jsonl"))


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


def encode_all(torch, encoder, rows: list[dict[str, Any]], batch_size: int, device: str):
    chunks = []
    for start in range(0, len(rows), batch_size):
        batch = batch_graphs(torch, rows, start, batch_size, device)
        chunks.append(encoder(batch))
    return torch.cat(chunks, dim=0)
