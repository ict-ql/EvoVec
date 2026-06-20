#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z_.$][A-Za-z0-9_.$]*|\d+")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_path(split_dir: Path, split: str) -> Path:
    return split_dir / f"{split}.jsonl"


def label_path(label_dir: Path, split: str) -> Path:
    return label_dir / f"{split}.labels.jsonl"


def record_key(row: dict[str, Any]) -> str:
    return str(row.get("dataset_index"))


def tokenize(text: str, max_tokens: int) -> list[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    if max_tokens > 0:
        tokens = tokens[:max_tokens]
    return tokens


def ir_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None and isinstance(row.get("ir"), dict):
        value = row["ir"].get(field)
    return str(value or "")


def load_joined(split_dir: Path, label_dir: Path, split: str, ir_field: str) -> list[dict[str, Any]]:
    split_rows = {record_key(row): row for row in load_jsonl(split_path(split_dir, split))}
    label_rows = load_jsonl(label_path(label_dir, split))
    joined: list[dict[str, Any]] = []
    for label in label_rows:
        if not label.get("ok"):
            continue
        key = record_key(label)
        source = split_rows.get(key)
        if source is None:
            continue
        instructions = [str(item) for item in label.get("instructions", [])]
        if not instructions:
            continue
        joined.append(
            {
                "dataset_index": key,
                "text": ir_text(source, ir_field),
                "instructions": instructions,
            }
        )
    return joined


def train_model(rows: list[dict[str, Any]], max_tokens: int, min_token_df: int, alpha: float) -> dict[str, Any]:
    token_df: Counter[str] = Counter()
    instruction_df: Counter[str] = Counter()
    instruction_token_df: dict[str, Counter[str]] = defaultdict(Counter)

    token_sets: list[tuple[set[str], set[str]]] = []
    for row in rows:
        tokens = set(tokenize(row["text"], max_tokens))
        instructions = set(row["instructions"])
        if not tokens or not instructions:
            continue
        token_sets.append((tokens, instructions))
        token_df.update(tokens)
        instruction_df.update(instructions)

    kept_tokens = {token for token, count in token_df.items() if count >= min_token_df}
    for tokens, instructions in token_sets:
        filtered = tokens & kept_tokens
        if not filtered:
            continue
        for instruction in instructions:
            instruction_token_df[instruction].update(filtered)

    n_docs = max(1, len(token_sets))
    instruction_vocab = sorted(instruction_df)
    priors = {
        instruction: math.log((instruction_df[instruction] + alpha) / (n_docs + alpha * len(instruction_vocab)))
        for instruction in instruction_vocab
    }
    token_weights: dict[str, dict[str, float]] = {}
    for instruction in instruction_vocab:
        denom = instruction_df[instruction] + 2.0 * alpha
        weights = {
            token: math.log((count + alpha) / denom)
            for token, count in instruction_token_df[instruction].items()
        }
        token_weights[instruction] = weights

    return {
        "kind": "token_instruction_naive_bayes",
        "alpha": alpha,
        "max_tokens": max_tokens,
        "min_token_df": min_token_df,
        "train_docs": len(token_sets),
        "instruction_vocab": instruction_vocab,
        "token_vocab_size": len(kept_tokens),
        "priors": priors,
        "token_weights": token_weights,
    }


def rank_instructions(model: dict[str, Any], text: str, top_k: int) -> list[tuple[str, float]]:
    tokens = set(tokenize(text, int(model["max_tokens"])))
    scores: list[tuple[str, float]] = []
    for instruction in model["instruction_vocab"]:
        score = float(model["priors"].get(instruction, -100.0))
        weights = model["token_weights"].get(instruction, {})
        for token in tokens:
            score += float(weights.get(token, 0.0))
        scores.append((instruction, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]


def evaluate(model: dict[str, Any], rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    totals = {k: 0.0 for k in ks}
    examples = 0
    max_k = max(ks)
    for row in rows:
        gold = set(row["instructions"])
        if not gold:
            continue
        ranked = [instruction for instruction, _score in rank_instructions(model, row["text"], max_k)]
        ranked_sets = {k: set(ranked[:k]) for k in ks}
        for k in ks:
            totals[k] += len(gold & ranked_sets[k]) / len(gold)
        examples += 1
    if examples == 0:
        return {"examples": 0, **{f"recall@{k}": 0.0 for k in ks}}
    return {
        "examples": examples,
        **{f"recall@{k}": totals[k] / examples for k in ks},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a simple IR-token to intrinsic instruction-label retrieval baseline."
    )
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/iroptset/models/baseline_pilot_10k_v1"))
    parser.add_argument("--ir-field", default="preprocessed_ir")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--min-token-df", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--eval-ks", default="10,20,50")
    args = parser.parse_args()

    train_rows = load_joined(args.split_dir, args.label_dir, "train", args.ir_field)
    if not train_rows:
        raise SystemExit("no joined train rows found; check split-dir, label-dir, and ir-field")
    model = train_model(train_rows, args.max_tokens, args.min_token_df, args.alpha)

    ks = [int(item) for item in args.eval_ks.split(",") if item.strip()]
    metrics = {"train": evaluate(model, train_rows, ks)}
    for split in ("val", "test"):
        if split_path(args.split_dir, split).exists() and label_path(args.label_dir, split).exists():
            rows = load_joined(args.split_dir, args.label_dir, split, args.ir_field)
            metrics[split] = evaluate(model, rows, ks)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
