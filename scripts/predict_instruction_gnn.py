#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_iroptset_graph_dataset import encode_graph, parse_ir_to_symbolic_graph
from train_instruction_gnn import RelationalGraphClassifier, batch_graphs, require_torch


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_output(text: str, out_path: Path | None) -> None:
    if out_path is None:
        print(text)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + ("\n" if text and not text.endswith("\n") else ""), encoding="utf-8")


def row_ir(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None and isinstance(row.get("ir"), dict):
        value = row["ir"].get(field)
    return str(value or "")


def load_checkpoint(torch, path: Path, device: str) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"checkpoint {path} did not contain a dictionary")
    for key in ("model_state", "config", "vocab"):
        if key not in checkpoint:
            raise SystemExit(f"checkpoint {path} is missing {key!r}")
    return checkpoint


def graph_row_from_ir(ir: str, vocab: dict[str, Any]) -> dict[str, Any]:
    params = vocab.get("params", {})
    symbolic = parse_ir_to_symbolic_graph(
        ir,
        int(params.get("max_nodes", 512)),
        int(params.get("max_edges", 4096)),
        int(params.get("max_operand_nodes", 128)),
    )
    encoded = encode_graph(symbolic, vocab["token_to_id"])
    return {
        "node_token_ids": encoded["node_token_ids"],
        "node_type_ids": encoded["node_type_ids"],
        "edges": encoded["edges"],
        "label_ids": [],
        "num_nodes": len(encoded["node_token_ids"]),
        "num_edges": len(encoded["edges"]),
    }


def build_model(torch, nn, F, checkpoint: dict[str, Any], device: str):
    config = checkpoint["config"]
    vocab = checkpoint["vocab"]
    wrapper = RelationalGraphClassifier(
        torch,
        nn,
        F,
        token_vocab_size=len(vocab["token_to_id"]),
        node_type_count=len(vocab["node_type_to_id"]),
        edge_type_count=len(vocab["edge_type_to_id"]),
        num_labels=len(vocab["instruction_to_id"]),
        hidden_dim=int(config["hidden_dim"]),
        layers=int(config["layers"]),
        dropout=float(config.get("dropout", 0.0)),
    )
    model = wrapper.model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def predict_one(torch, model, graph_row: dict[str, Any], vocab: dict[str, Any], top_k: int, device: str) -> dict[str, Any]:
    instruction_to_id = vocab["instruction_to_id"]
    id_to_instruction = {int(label_id): instruction for instruction, label_id in instruction_to_id.items()}
    num_labels = len(instruction_to_id)
    batch = batch_graphs([graph_row], 0, 1, num_labels, device)
    with torch.no_grad():
        logits = model(batch)[0]
        probs = torch.sigmoid(logits)
        k = min(top_k, num_labels)
        values, indices = torch.topk(logits, k=k)
    predictions = []
    for rank, (label_id, score) in enumerate(zip(indices.cpu().tolist(), values.cpu().tolist()), start=1):
        predictions.append(
            {
                "rank": rank,
                "instruction": id_to_instruction[int(label_id)],
                "score": float(score),
                "probability": float(probs[int(label_id)].cpu().item()),
            }
        )
    return {
        "num_nodes": graph_row["num_nodes"],
        "num_edges": graph_row["num_edges"],
        "top_k": k,
        "instructions": predictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Use a trained instruction GNN checkpoint to rank x86 instruction mnemonics for LLVM IR."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ir-file", type=Path, help="LLVM IR file to score. Use '-' to read stdin.")
    parser.add_argument("--jsonl", type=Path, help="Optional JSONL input with an IR field; writes one JSON object per row.")
    parser.add_argument("--ir-field", default="preprocessed_ir")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--format", choices=["json", "palette"], default="json")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if bool(args.ir_file) == bool(args.jsonl):
        parser.error("provide exactly one of --ir-file or --jsonl")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.format == "palette" and args.jsonl:
        parser.error("--format palette is only supported with --ir-file")

    torch, nn, F = require_torch()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    checkpoint = load_checkpoint(torch, args.checkpoint, device)
    vocab = checkpoint["vocab"]
    model = build_model(torch, nn, F, checkpoint, device)

    if args.ir_file:
        if str(args.ir_file) == "-":
            import sys

            ir = sys.stdin.read()
        else:
            ir = args.ir_file.read_text(encoding="utf-8")
        graph_row = graph_row_from_ir(ir, vocab)
        result = {
            "checkpoint": str(args.checkpoint),
            **predict_one(torch, model, graph_row, vocab, args.top_k, device),
        }
        if args.format == "palette":
            text = ", ".join(item["instruction"] for item in result["instructions"])
        else:
            text = json.dumps(result, indent=2, sort_keys=True)
        write_output(text, args.out)
        return 0

    assert args.jsonl is not None
    outputs = []
    for row in load_jsonl(args.jsonl):
        ir = row_ir(row, args.ir_field)
        graph_row = graph_row_from_ir(ir, vocab)
        result = predict_one(torch, model, graph_row, vocab, args.top_k, device)
        outputs.append(
            {
                "dataset_index": row.get("dataset_index"),
                "repo_name": row.get("repo_name"),
                "repo_file_path": row.get("repo_file_path"),
                "function_name": row.get("function_name"),
                **result,
            }
        )
    text = "\n".join(json.dumps(row, sort_keys=True) for row in outputs)
    write_output(text, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
