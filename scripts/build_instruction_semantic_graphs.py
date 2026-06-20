#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z_.$][A-Za-z0-9_.$]*|[-+]?\d+")
ASSIGN_RE = re.compile(r"(?P<lhs>.+?)(?::=|=)(?P<rhs>.+)")

NODE_TYPES = {
    "instruction": 0,
    "intrinsic": 1,
    "assembly": 2,
    "architecture": 3,
    "extension": 4,
    "cpuid": 5,
    "operand": 6,
    "type": 7,
    "micro_op": 8,
    "operator": 9,
    "value": 10,
    "constant": 11,
    "predicate": 12,
}

EDGE_TYPES = {
    "has_intrinsic": 0,
    "lowers_to": 1,
    "belongs_to": 2,
    "requires": 3,
    "has_operand": 4,
    "operand_type": 5,
    "defines": 6,
    "reads": 7,
    "semantic_step": 8,
    "next_step": 9,
    "computes": 10,
    "has_value": 11,
    "lhs": 12,
    "rhs": 13,
    "controls": 14,
}

OPERATOR_PATTERNS = (
    ("zero_extend", re.compile(r"\b(?:zeroextend|zext|zero_extend)\b", re.IGNORECASE)),
    ("sign_extend", re.compile(r"\b(?:signextend|sext|sign_extend)\b", re.IGNORECASE)),
    ("saturate", re.compile(r"\b(?:saturate|sat)\b", re.IGNORECASE)),
    ("sum", re.compile(r"\b(?:sum|reduce_add)\b", re.IGNORECASE)),
    ("mask", re.compile(r"\bmask\b|{k}", re.IGNORECASE)),
    ("add", re.compile(r"\+|\badd\b", re.IGNORECASE)),
    ("sub", re.compile(r"(?<![A-Za-z0-9_])-|\bsub\b", re.IGNORECASE)),
    ("mul", re.compile(r"\*|\bmul\b", re.IGNORECASE)),
    ("shift", re.compile(r"<<|>>|\bshift\b", re.IGNORECASE)),
    ("compare", re.compile(r"==|!=|<=|>=|<|>|\bcompare\b", re.IGNORECASE)),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def normalize_token(token: str) -> str:
    token = token.strip().lower().strip(",()[]{}")
    if not token:
        return "empty"
    if re.fullmatch(r"[-+]?\d+", token):
        return f"const_{token.replace('-', 'neg')}"
    return token


def first_text(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value:
            return str(value)
    return ""


def instruction_label(record: dict[str, Any]) -> str:
    label = first_text(record, "label", "llvm_intrinsic", "intrinsic", "instruction")
    if not label:
        raise ValueError("instruction spec is missing one of label/llvm_intrinsic/intrinsic/instruction")
    return label


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def operation_lines(record: dict[str, Any]) -> list[str]:
    value = record.get("operation", [])
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return [str(line).strip() for line in as_list(value) if str(line).strip()]


def classify_micro_op(line: str) -> str:
    lowered = line.strip().lower()
    if lowered.startswith(("for ", "foreach ", "while ")):
        return "loop"
    if lowered.startswith(("if ", "when ")) or " mask" in lowered:
        return "predicate"
    if lowered in {"endfor", "endif", "end"}:
        return "end"
    if ASSIGN_RE.search(line):
        return "assign"
    for operator, pattern in OPERATOR_PATTERNS:
        if pattern.search(line):
            return operator
    return "effect"


def mentioned_operands(text: str, operand_names: set[str]) -> set[str]:
    tokens: set[str] = set()
    for token in TOKEN_RE.findall(text):
        normalized = normalize_token(token)
        tokens.add(normalized)
        if "." in normalized:
            tokens.add(normalized.split(".", 1)[0])
    return tokens & operand_names


class GraphBuilder:
    def __init__(self) -> None:
        self.node_tokens: list[str] = []
        self.node_types: list[str] = []
        self.edges: list[tuple[int, int, str]] = []

    def add_node(self, token: str, node_type: str) -> int:
        self.node_tokens.append(normalize_token(token))
        self.node_types.append(node_type)
        return len(self.node_tokens) - 1

    def add_edge(self, src: int, dst: int, edge_type: str) -> None:
        self.edges.append((src, dst, edge_type))


def add_optional_metadata(builder: GraphBuilder, root: int, record: dict[str, Any]) -> None:
    intrinsic = first_text(record, "llvm_intrinsic", "intrinsic")
    if intrinsic:
        node = builder.add_node(intrinsic, "intrinsic")
        builder.add_edge(root, node, "has_intrinsic")

    assembly = first_text(record, "instruction", "assembly")
    if assembly:
        node = builder.add_node(assembly.split()[0], "assembly")
        builder.add_edge(root, node, "lowers_to")

    architecture = first_text(record, "architecture", "isa")
    if architecture:
        node = builder.add_node(architecture, "architecture")
        builder.add_edge(root, node, "belongs_to")

    extension = first_text(record, "extension", "instruction_set")
    if extension:
        node = builder.add_node(extension, "extension")
        builder.add_edge(root, node, "belongs_to")

    for flag in as_list(record.get("cpuid")):
        node = builder.add_node(str(flag), "cpuid")
        builder.add_edge(root, node, "requires")


def add_operands(builder: GraphBuilder, root: int, record: dict[str, Any]) -> dict[str, int]:
    operand_nodes: dict[str, int] = {}
    for operand in as_list(record.get("operands")):
        if not isinstance(operand, dict):
            continue
        name = normalize_token(str(operand.get("name", "")))
        if not name:
            continue
        operand_node = builder.add_node(name, "operand")
        builder.add_edge(root, operand_node, "has_operand")
        operand_nodes[name] = operand_node
        operand_type = operand.get("type")
        if operand_type:
            type_node = builder.add_node(str(operand_type), "type")
            builder.add_edge(operand_node, type_node, "operand_type")
    return operand_nodes


def add_line_values(builder: GraphBuilder, micro_node: int, text: str) -> None:
    for token in TOKEN_RE.findall(text):
        normalized = normalize_token(token)
        if normalized.startswith("const_"):
            value_node = builder.add_node(normalized, "constant")
            builder.add_edge(micro_node, value_node, "has_value")


def add_operator_nodes(builder: GraphBuilder, micro_node: int, line: str) -> None:
    seen: set[str] = set()
    for operator, pattern in OPERATOR_PATTERNS:
        if operator not in seen and pattern.search(line):
            node = builder.add_node(operator, "operator")
            builder.add_edge(micro_node, node, "computes")
            seen.add(operator)


def add_operand_edges(
    builder: GraphBuilder,
    micro_node: int,
    line: str,
    operand_nodes: dict[str, int],
) -> None:
    if not operand_nodes:
        return
    match = ASSIGN_RE.search(line)
    if match:
        lhs_mentions = mentioned_operands(match.group("lhs"), set(operand_nodes))
        rhs_mentions = mentioned_operands(match.group("rhs"), set(operand_nodes))
        for name in sorted(lhs_mentions):
            builder.add_edge(micro_node, operand_nodes[name], "defines")
            builder.add_edge(micro_node, operand_nodes[name], "lhs")
        for name in sorted(rhs_mentions):
            builder.add_edge(micro_node, operand_nodes[name], "reads")
            builder.add_edge(micro_node, operand_nodes[name], "rhs")
        return

    for name in sorted(mentioned_operands(line, set(operand_nodes))):
        builder.add_edge(micro_node, operand_nodes[name], "reads")


def build_symbolic_graph(record: dict[str, Any]) -> dict[str, Any]:
    builder = GraphBuilder()
    label = instruction_label(record)
    root = builder.add_node(label, "instruction")
    add_optional_metadata(builder, root, record)
    operand_nodes = add_operands(builder, root, record)

    previous_micro_op: int | None = None
    active_control: int | None = None
    for line in operation_lines(record):
        micro_op = classify_micro_op(line)
        micro_node = builder.add_node(f"micro_{micro_op}", "micro_op")
        builder.add_edge(root, micro_node, "semantic_step")
        if previous_micro_op is not None:
            builder.add_edge(previous_micro_op, micro_node, "next_step")
        if active_control is not None and micro_op != "end":
            builder.add_edge(active_control, micro_node, "controls")
        if micro_op in {"loop", "predicate"}:
            active_control = micro_node
        elif micro_op == "end":
            active_control = None
        add_operator_nodes(builder, micro_node, line)
        add_line_values(builder, micro_node, line)
        add_operand_edges(builder, micro_node, line, operand_nodes)
        previous_micro_op = micro_node

    return {
        "node_tokens": builder.node_tokens,
        "node_types": builder.node_types,
        "edges": builder.edges,
    }


def build_vocab(counter: Counter[str], max_size: int, min_count: int) -> dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in counter.most_common():
        if count < min_count:
            continue
        if token in vocab:
            continue
        if max_size > 0 and len(vocab) >= max_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_graph(symbolic: dict[str, Any], token_to_id: dict[str, int]) -> dict[str, Any]:
    return {
        "node_token_ids": [token_to_id.get(token, token_to_id["<unk>"]) for token in symbolic["node_tokens"]],
        "node_type_ids": [NODE_TYPES[item] for item in symbolic["node_types"]],
        "edges": [[src, dst, EDGE_TYPES[edge_type]] for src, dst, edge_type in symbolic["edges"]],
    }


def make_output_record(
    record: dict[str, Any],
    symbolic: dict[str, Any],
    token_to_id: dict[str, int],
    include_symbols: bool,
) -> dict[str, Any]:
    encoded = encode_graph(symbolic, token_to_id)
    output = {
        "label": instruction_label(record),
        "split": str(record.get("split", "train")),
        "architecture": first_text(record, "architecture", "isa"),
        "extension": first_text(record, "extension", "instruction_set"),
        "intrinsic": first_text(record, "llvm_intrinsic", "intrinsic"),
        "instruction": first_text(record, "instruction", "assembly"),
        "num_nodes": len(encoded["node_token_ids"]),
        "num_edges": len(encoded["edges"]),
        "node_token_ids": encoded["node_token_ids"],
        "node_type_ids": encoded["node_type_ids"],
        "edges": encoded["edges"],
    }
    if include_symbols:
        output["node_tokens"] = symbolic["node_tokens"]
        output["node_types"] = symbolic["node_types"]
        output["symbolic_edges"] = symbolic["edges"]
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build instruction semantic graphs from intrinsic/instruction micro-op specs."
    )
    parser.add_argument("--spec-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/instruction_semantics/sample"))
    parser.add_argument("--max-token-vocab", type=int, default=20000)
    parser.add_argument("--min-token-count", type=int, default=1)
    parser.add_argument("--drop-symbols", action="store_true", help="Do not store node tokens/types in graph rows.")
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for path in args.spec_jsonl:
        records.extend(load_jsonl(path))
    if not records:
        raise SystemExit("no instruction specs found")

    symbolic_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    token_counter: Counter[str] = Counter()
    for record in records:
        symbolic = build_symbolic_graph(record)
        symbolic_rows.append((record, symbolic))
        token_counter.update(symbolic["node_tokens"])

    token_to_id = build_vocab(token_counter, args.max_token_vocab, args.min_token_count)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vocab = {
        "token_to_id": token_to_id,
        "node_type_to_id": NODE_TYPES,
        "edge_type_to_id": EDGE_TYPES,
        "params": {
            "max_token_vocab": args.max_token_vocab,
            "min_token_count": args.min_token_count,
        },
    }
    (args.out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    summary: dict[str, Any] = {"splits": {}, "vocab": str(args.out_dir / "vocab.json")}
    for record, symbolic in symbolic_rows:
        split = str(record.get("split", "train"))
        by_split[split].append(
            make_output_record(record, symbolic, token_to_id, include_symbols=not args.drop_symbols)
        )

    for split, rows in sorted(by_split.items()):
        out_path = args.out_dir / f"{split}.instruction_graphs.jsonl"
        write_jsonl(out_path, rows)
        summary["splits"][split] = {
            "rows": len(rows),
            "path": str(out_path),
            "architectures": sorted({row["architecture"] for row in rows if row["architecture"]}),
            "extensions": sorted({row["extension"] for row in rows if row["extension"]}),
        }
        print(json.dumps({"split": split, **summary["splits"][split]}, sort_keys=True))

    summary["token_vocab_size"] = len(token_to_id)
    summary["node_type_to_id"] = NODE_TYPES
    summary["edge_type_to_id"] = EDGE_TYPES
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
