#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any


TOKEN_RE = re.compile(r"[%@]?[A-Za-z_.$][A-Za-z0-9_.$]*|[-+]?\d+(?:\.\d+)?")
SSA_RE = re.compile(r"%[A-Za-z0-9_.$-]+")
GLOBAL_RE = re.compile(r"@[A-Za-z0-9_.$-]+")
LABEL_REF_RE = re.compile(r"\blabel\s+%([A-Za-z0-9_.$-]+)")

NODE_TYPES = {
    "function": 0,
    "block": 1,
    "instruction": 2,
    "argument": 3,
    "operand": 4,
}
EDGE_TYPES = {
    "contains": 0,
    "contained_by": 1,
    "next": 2,
    "prev": 3,
    "def_use": 4,
    "use_def": 5,
    "operand": 6,
    "used_by": 7,
    "branch": 8,
    "branch_rev": 9,
}

LLVM_PREFIXES = {
    "tail",
    "musttail",
    "notail",
    "fast",
    "nuw",
    "nsw",
    "exact",
    "inbounds",
    "atomic",
    "volatile",
}


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


def split_path(split_dir: Path, split: str) -> Path:
    return split_dir / f"{split}.jsonl"


def label_path(label_dir: Path, split: str) -> Path:
    return label_dir / f"{split}.labels.jsonl"


def record_key(row: dict[str, Any]) -> str:
    return str(row.get("dataset_index"))


def ir_text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None and isinstance(row.get("ir"), dict):
        value = row["ir"].get(field)
    return str(value or "")


def strip_comment(line: str) -> str:
    if ";" in line:
        line = line.split(";", 1)[0]
    return line.strip()


def normalize_token(token: str) -> str:
    token = token.strip().lower().strip(",()[]{}")
    if not token:
        return "empty"
    if token.startswith("%"):
        return "ssa_value"
    if token.startswith("@"):
        return "global_value"
    if token in {"null", "undef", "poison", "zeroinitializer"}:
        return f"const_{token}"
    if re.fullmatch(r"[-+]?\d+", token):
        if token in {"0", "1", "-1", "2", "4", "8", "16", "32", "64"}:
            return f"const_{token.replace('-', 'neg')}"
        return "const_int"
    if re.fullmatch(r"[-+]?\d+\.\d+", token):
        return "const_float"
    return token


def instruction_opcode(text: str) -> str:
    rhs = text.split("=", 1)[1].strip() if "=" in text else text.strip()
    tokens = [normalize_token(tok) for tok in TOKEN_RE.findall(rhs)]
    for token in tokens:
        if token in LLVM_PREFIXES:
            continue
        if token.startswith("const_"):
            continue
        if token in {"void", "i1", "i8", "i16", "i32", "i64", "float", "double", "ptr"}:
            continue
        return token
    return "unknown_op"


class GraphBuilder:
    def __init__(self, max_nodes: int, max_edges: int, max_operand_nodes: int) -> None:
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_operand_nodes = max_operand_nodes
        self.node_tokens: list[str] = []
        self.node_types: list[str] = []
        self.edges: list[tuple[int, int, str]] = []
        self.defs: dict[str, int] = {}
        self.blocks: dict[str, int] = {}
        self.current_block: int | None = None
        self.last_inst: int | None = None
        self.operand_nodes = 0

    def add_node(self, token: str, node_type: str) -> int | None:
        if len(self.node_tokens) >= self.max_nodes:
            return None
        self.node_tokens.append(normalize_token(token))
        self.node_types.append(node_type)
        return len(self.node_tokens) - 1

    def add_edge(self, src: int | None, dst: int | None, edge_type: str) -> None:
        if src is None or dst is None:
            return
        if len(self.edges) >= self.max_edges:
            return
        self.edges.append((src, dst, edge_type))

    def add_bidirectional(self, src: int | None, dst: int | None, fwd: str, rev: str) -> None:
        self.add_edge(src, dst, fwd)
        self.add_edge(dst, src, rev)

    def add_operand_node(self, token: str, inst: int | None) -> None:
        if inst is None or self.operand_nodes >= self.max_operand_nodes:
            return
        node = self.add_node(token, "operand")
        if node is None:
            return
        self.operand_nodes += 1
        self.add_bidirectional(node, inst, "used_by", "operand")

    def define_args_from_header(self, header: str) -> None:
        for arg in SSA_RE.findall(header):
            if arg in self.defs:
                continue
            node = self.add_node("arg", "argument")
            self.defs[arg] = node if node is not None else -1

    def add_block(self, name: str, root: int | None) -> None:
        block = self.add_node(f"block_{name}", "block")
        if block is None:
            return
        self.blocks[name] = block
        self.current_block = block
        self.last_inst = None
        self.add_bidirectional(root, block, "contains", "contained_by")

    def add_instruction(self, line: str) -> None:
        op = instruction_opcode(line)
        inst = self.add_node(op, "instruction")
        if inst is None:
            return
        if self.current_block is not None:
            self.add_bidirectional(self.current_block, inst, "contains", "contained_by")
        if self.last_inst is not None:
            self.add_bidirectional(self.last_inst, inst, "next", "prev")
        self.last_inst = inst

        lhs = None
        if "=" in line:
            lhs_text = line.split("=", 1)[0].strip()
            match = SSA_RE.search(lhs_text)
            if match:
                lhs = match.group(0)
                self.defs[lhs] = inst

        uses = set(SSA_RE.findall(line))
        if lhs:
            uses.discard(lhs)
        for use in sorted(uses):
            src = self.defs.get(use)
            if src is not None and src >= 0:
                self.add_bidirectional(src, inst, "def_use", "use_def")
            else:
                self.add_operand_node("ssa_value", inst)

        for global_name in sorted(set(GLOBAL_RE.findall(line))):
            self.add_operand_node(global_name, inst)

        for label in LABEL_REF_RE.findall(line):
            block = self.blocks.get(label)
            if block is not None:
                self.add_bidirectional(inst, block, "branch", "branch_rev")


def parse_ir_to_symbolic_graph(ir: str, max_nodes: int, max_edges: int, max_operand_nodes: int) -> dict[str, Any]:
    builder = GraphBuilder(max_nodes, max_edges, max_operand_nodes)
    root = builder.add_node("function", "function")
    in_body = False
    saw_block = False
    for raw_line in ir.splitlines():
        line = strip_comment(raw_line)
        if not line:
            continue
        if line.startswith("define "):
            builder.define_args_from_header(line)
            in_body = "{" in line
            continue
        if not in_body:
            continue
        if line == "}":
            break
        if line.endswith(":"):
            label = line[:-1].strip().lstrip("%")
            builder.add_block(label, root)
            saw_block = True
            continue
        if not saw_block:
            builder.add_block("entry", root)
            saw_block = True
        builder.add_instruction(line)

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


def load_joined(split_dir: Path, label_dir: Path, split: str, ir_field: str) -> list[dict[str, Any]]:
    source_rows = {record_key(row): row for row in load_jsonl(split_path(split_dir, split))}
    label_rows = load_jsonl(label_path(label_dir, split))
    rows: list[dict[str, Any]] = []
    for label in label_rows:
        if not label.get("ok"):
            continue
        key = record_key(label)
        source = source_rows.get(key)
        if source is None:
            continue
        instructions = [str(item) for item in label.get("instructions", [])]
        if not instructions:
            continue
        rows.append(
            {
                "dataset_index": key,
                "repo_name": source.get("repo_name"),
                "repo_file_path": source.get("repo_file_path"),
                "function_name": source.get("function_name"),
                "structural_hash": source.get("structural_hash"),
                "ir": ir_text(source, ir_field),
                "instructions": instructions,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build graph JSONL records from IR-OptSet splits and instruction labels."
    )
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/iroptset/graphs/pilot_10k_v1"))
    parser.add_argument("--ir-field", default="preprocessed_ir")
    parser.add_argument("--max-token-vocab", type=int, default=20000)
    parser.add_argument("--min-token-count", type=int, default=2)
    parser.add_argument("--max-instruction-vocab", type=int, default=512)
    parser.add_argument("--min-instruction-count", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=512)
    parser.add_argument("--max-edges", type=int, default=4096)
    parser.add_argument("--max-operand-nodes", type=int, default=128)
    args = parser.parse_args()

    train_rows = load_joined(args.split_dir, args.label_dir, "train", args.ir_field)
    if not train_rows:
        raise SystemExit("no joined train rows found")

    token_counter: Counter[str] = Counter()
    instruction_counter: Counter[str] = Counter()
    train_symbolic: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        graph = parse_ir_to_symbolic_graph(
            row["ir"], args.max_nodes, args.max_edges, args.max_operand_nodes
        )
        train_symbolic[row["dataset_index"]] = graph
        token_counter.update(graph["node_tokens"])
        instruction_counter.update(row["instructions"])

    token_to_id = build_vocab(token_counter, args.max_token_vocab, args.min_token_count)
    instruction_to_id = build_vocab(
        instruction_counter, args.max_instruction_vocab + 2, args.min_instruction_count
    )
    instruction_to_id.pop("<pad>", None)
    instruction_to_id.pop("<unk>", None)
    instruction_to_id = {instruction: index for index, instruction in enumerate(instruction_to_id)}
    if not instruction_to_id:
        raise SystemExit("instruction vocabulary is empty")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vocab = {
        "token_to_id": token_to_id,
        "node_type_to_id": NODE_TYPES,
        "edge_type_to_id": EDGE_TYPES,
        "instruction_to_id": instruction_to_id,
        "params": {
            "ir_field": args.ir_field,
            "max_nodes": args.max_nodes,
            "max_edges": args.max_edges,
            "max_operand_nodes": args.max_operand_nodes,
            "max_token_vocab": args.max_token_vocab,
            "min_token_count": args.min_token_count,
            "max_instruction_vocab": args.max_instruction_vocab,
            "min_instruction_count": args.min_instruction_count,
        },
    }
    (args.out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    summary: dict[str, Any] = {"splits": {}, "vocab": str(args.out_dir / "vocab.json")}
    for split in ("train", "val", "test"):
        if not split_path(args.split_dir, split).exists() or not label_path(args.label_dir, split).exists():
            continue
        rows = load_joined(args.split_dir, args.label_dir, split, args.ir_field)
        output_rows: list[dict[str, Any]] = []
        skipped_no_label = 0
        for row in rows:
            graph = train_symbolic.get(row["dataset_index"])
            if graph is None:
                graph = parse_ir_to_symbolic_graph(
                    row["ir"], args.max_nodes, args.max_edges, args.max_operand_nodes
                )
            label_ids = sorted(
                {
                    instruction_to_id[item]
                    for item in row["instructions"]
                    if item in instruction_to_id
                }
            )
            if not label_ids:
                skipped_no_label += 1
                continue
            encoded = encode_graph(graph, token_to_id)
            output_rows.append(
                {
                    "dataset_index": row["dataset_index"],
                    "repo_name": row.get("repo_name"),
                    "repo_file_path": row.get("repo_file_path"),
                    "function_name": row.get("function_name"),
                    "structural_hash": row.get("structural_hash"),
                    "num_nodes": len(encoded["node_token_ids"]),
                    "num_edges": len(encoded["edges"]),
                    "node_token_ids": encoded["node_token_ids"],
                    "node_type_ids": encoded["node_type_ids"],
                    "edges": encoded["edges"],
                    "label_ids": label_ids,
                    "instructions": [item for item in row["instructions"] if item in instruction_to_id],
                }
            )
        out_path = args.out_dir / f"{split}.graphs.jsonl"
        write_jsonl(out_path, output_rows)
        summary["splits"][split] = {
            "rows": len(output_rows),
            "skipped_no_label": skipped_no_label,
            "path": str(out_path),
        }
        print(json.dumps({"split": split, **summary["splits"][split]}, sort_keys=True))

    summary["token_vocab_size"] = len(token_to_id)
    summary["instruction_vocab_size"] = len(instruction_to_id)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
