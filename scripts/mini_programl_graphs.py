#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

try:
    from vegen_operation_parser import (
        BinaryExpr,
        BitSlice,
        Call,
        For,
        If,
        Lookup,
        Number,
        PseudoStmt,
        UnaryExpr,
        Update,
        Var,
        ast_to_json,
        parse_operation_ast,
    )
except ModuleNotFoundError:  # pragma: no cover - used when imported as a package module
    from scripts.vegen_operation_parser import (
        BinaryExpr,
        BitSlice,
        Call,
        For,
        If,
        Lookup,
        Number,
        PseudoStmt,
        UnaryExpr,
        Update,
        Var,
        ast_to_json,
        parse_operation_ast,
    )


NODE_TYPES = {
    "instruction": 0,
    "variable": 1,
    "constant": 2,
    "type": 3,
}

EDGE_TYPES = {
    "control": 0,
    "data": 1,
    "type": 2,
}

EDGE_FEATURE_TYPES = {
    "none": 0,
    "control_backedge": 1,
    "control_loop_self": 2,
    "control_loop_header": 3,
    "control_loop_body": 4,
    "control_predicate": 5,
    "control_scope": 6,
    "role_arg": 7,
    "role_base": 8,
    "role_constant": 9,
    "role_def": 10,
    "role_def_base": 11,
    "role_global": 12,
    "role_hi": 13,
    "role_induction_def": 14,
    "role_lhs": 15,
    "role_lo": 16,
    "role_lower_bound": 17,
    "role_operand": 18,
    "role_predicate": 19,
    "role_rhs": 20,
    "role_upper_bound": 21,
    "role_use": 22,
    "role_value": 23,
    "role_loop_domain": 24,
}

GRAPH_SCHEMA = "mini-programl"

TOKEN_RE = re.compile(r"[%@]?[A-Za-z_.$][A-Za-z0-9_.$-]*|[-+]?\d+(?:\.\d+)?")
SSA_RE = re.compile(r"%[A-Za-z0-9_.$-]+")
GLOBAL_RE = re.compile(r"@[A-Za-z0-9_.$-]+")
LABEL_REF_RE = re.compile(r"\blabel\s+%([A-Za-z0-9_.$-]+)")
PHI_INCOMING_RE = re.compile(r"\[\s*(?P<value>[^,\]]+?)\s*,\s*%(?P<block>[A-Za-z0-9_.$-]+)\s*\]")
ASSIGN_RE = re.compile(r"(?P<lhs>.+?)(?::=|=)(?P<rhs>.+)")
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NUMBER_RE = re.compile(r"[-+]?\d+")
FOR_ASSIGN_RE = re.compile(
    r"^\s*for\s+(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*(?::=|=)\s*"
    r"(?P<start>[-+]?\d+)\s+to\s+(?P<end>[-+]?\d+)",
    re.IGNORECASE,
)
FOR_RANGE_RE = re.compile(
    r"^\s*for\s+(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s+in\s+"
    r"(?P<start>[-+]?\d+)\s*(?:\.\.|to)\s*(?P<end>[-+]?\d+)",
    re.IGNORECASE,
)
FOR_DOMAIN_RE = re.compile(
    r"^\s*(?:for|foreach)\s+(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s+in\s+(?P<domain>.+)",
    re.IGNORECASE,
)

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

LLVM_TYPE_ATOMS = {
    "void",
    "label",
    "metadata",
    "token",
    "ptr",
    "half",
    "bfloat",
    "float",
    "double",
    "x86_fp80",
    "fp128",
    "ppc_fp128",
}

LLVM_TYPE_PATTERN = (
    r"(?:<\s*\d+\s+x\s+(?:ptr|i\d+|half|bfloat|float|double|x86_fp80|fp128|ppc_fp128)\s*>"
    r"|(?:ptr|void|label|metadata|token|half|bfloat|float|double|x86_fp80|fp128|ppc_fp128|i\d+)(?:\s*\*)*)"
)
LLVM_TYPE_RE = re.compile(LLVM_TYPE_PATTERN)
LLVM_TYPED_VALUE_RE = re.compile(
    rf"(?P<type>{LLVM_TYPE_PATTERN})\s+"
    r"(?P<value>%[A-Za-z0-9_.$-]+|@[A-Za-z0-9_.$-]+|[-+]?\d+(?:\.\d+)?)"
)

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


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def first_text(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value:
            return str(value)
    return ""


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


def normalize_type_token(token: str) -> str:
    token = re.sub(r"\s+", "", token.strip().lower().rstrip(","))
    if not token:
        return "empty"
    pointer_depth = 0
    while token.endswith("*"):
        pointer_depth += 1
        token = token[:-1]
    vector_match = re.fullmatch(r"<(\d+)x(.+)>", token)
    if vector_match:
        token = f"v{vector_match.group(1)}x{normalize_type_token(vector_match.group(2))}"
    else:
        token = normalize_token(token)
    if pointer_depth:
        token += "_ptr" * pointer_depth
    return token


def is_llvm_type_token(token: str) -> bool:
    normalized = normalize_type_token(token)
    return (
        normalized in LLVM_TYPE_ATOMS
        or re.fullmatch(r"i\d+", normalized) is not None
        or re.fullmatch(r"v\d+x.+", normalized) is not None
        or normalized.endswith("_ptr")
    )


def first_llvm_type(text: str) -> str | None:
    match = LLVM_TYPE_RE.search(text)
    if not match:
        return None
    return normalize_type_token(match.group(0))


def typed_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for match in LLVM_TYPED_VALUE_RE.finditer(text):
        values[match.group("value")] = normalize_type_token(match.group("type"))
    return values


def merge_node_features(target: dict[str, Any], features: dict[str, Any] | None) -> None:
    if not features:
        return
    for key, value in features.items():
        if value in ("", None):
            continue
        if key == "data_type" and key in target and target[key] != value:
            observed = target.setdefault("observed_data_types", [])
            if isinstance(observed, list) and value not in observed:
                observed.append(value)
            continue
        target[key] = value


def instruction_label(record: dict[str, Any]) -> str:
    label = first_text(record, "label", "llvm_intrinsic", "intrinsic", "instruction")
    if not label:
        raise ValueError("instruction spec is missing one of label/llvm_intrinsic/intrinsic/instruction")
    return label


def record_aliases(record: dict[str, Any]) -> list[str]:
    aliases: list[str] = []
    for key in ("label", "llvm_intrinsic", "intrinsic", "instruction"):
        value = record.get(key)
        if value:
            aliases.append(str(value))
    aliases.extend(str(item) for item in as_list(record.get("aliases")) if str(item))
    return list(dict.fromkeys(aliases))


def operation_lines(record: dict[str, Any]) -> list[str]:
    value = record.get("operation", [])
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return [str(line).strip() for line in as_list(value) if str(line).strip()]


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
        "edges": [[src, dst, EDGE_TYPES[edge_type], position] for src, dst, edge_type, position in symbolic["edges"]],
        "edge_feature_ids": [EDGE_FEATURE_TYPES[edge_feature_label(item)] for item in symbolic.get("edge_features", [])],
        "edge_counts": [edge_count_value(item) for item in symbolic.get("edge_features", [])],
    }


def edge_feature_label(features: dict[str, Any]) -> str:
    role = features.get("role")
    if role:
        label = "role_" + str(role)
        return label if label in EDGE_FEATURE_TYPES else "none"
    if features.get("loop_self"):
        return "control_loop_self"
    if features.get("backedge"):
        return "control_backedge"
    if features.get("loop_header"):
        return "control_loop_header"
    if features.get("predicate"):
        return "control_predicate"
    if features.get("scope") == "loop_body":
        return "control_loop_body"
    if features.get("scope"):
        return "control_scope"
    return "none"


def edge_count_value(features: dict[str, Any]) -> int:
    for key in ("execution_count", "iteration_count", "backedge_count"):
        value = features.get(key)
        if isinstance(value, int):
            return max(0, value)
    return 0


def compact_node_features(features: list[dict[str, Any]], include_debug_text: bool = False) -> list[dict[str, Any]]:
    debug_keys = {"full_text", "block_name"}
    if include_debug_text:
        return features
    return [{key: value for key, value in item.items() if key not in debug_keys} for item in features]


def compact_graph_features(features: dict[str, Any], include_debug_text: bool = False) -> dict[str, Any]:
    debug_keys = {"function_header", "operation_ast"}
    if include_debug_text:
        return features
    return {key: value for key, value in features.items() if key not in debug_keys}


class MiniProgramlBuilder:
    def __init__(
        self,
        max_nodes: int = 0,
        max_edges: int = 0,
        max_operand_nodes: int = 0,
    ) -> None:
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_operand_nodes = max_operand_nodes
        self.node_tokens: list[str] = []
        self.node_types: list[str] = []
        self.node_features: list[dict[str, Any]] = []
        self.edges: list[tuple[int, int, str, int]] = []
        self.edge_features: list[dict[str, Any]] = []
        self.defs: dict[str, int] = {}
        self.variable_nodes: dict[str, int] = {}
        self.type_nodes: dict[str, int] = {}
        self.type_annotations: set[tuple[int, str]] = set()
        self.operand_nodes = 0

    def add_node(self, token: str, node_type: str, features: dict[str, Any] | None = None) -> int | None:
        if self.max_nodes > 0 and len(self.node_tokens) >= self.max_nodes:
            return None
        features = dict(features or {})
        if node_type != "type":
            features.pop("data_type", None)
        self.node_tokens.append(normalize_token(token))
        self.node_types.append(node_type)
        self.node_features.append(features)
        return len(self.node_tokens) - 1

    def add_edge(
        self,
        src: int | None,
        dst: int | None,
        edge_type: str,
        position: int = 0,
        features: dict[str, Any] | None = None,
    ) -> int | None:
        if src is None or dst is None:
            return None
        if self.max_edges > 0 and len(self.edges) >= self.max_edges:
            return None
        self.edges.append((src, dst, edge_type, position))
        self.edge_features.append(features or {})
        return len(self.edges) - 1

    def variable_node(self, name: str, full_text: str | None = None, features: dict[str, Any] | None = None) -> int | None:
        if name in self.variable_nodes:
            node = self.variable_nodes[name]
            merge_node_features(self.node_features[node], features)
            return node
        node = self.add_node("var", "variable", {"full_text": full_text or name, **(features or {})})
        if node is not None:
            self.variable_nodes[name] = node
        return node

    def constant_node(
        self,
        token: str,
        full_text: str | None = None,
        features: dict[str, Any] | None = None,
    ) -> int | None:
        if self.max_operand_nodes > 0 and self.operand_nodes >= self.max_operand_nodes:
            return None
        node = self.add_node(
            "val",
            "constant",
            {
                "full_text": full_text or token,
                "value": str(full_text or token),
                **(features or {}),
            },
        )
        if node is not None:
            self.operand_nodes += 1
        return node

    def type_node(self, token: str) -> int | None:
        normalized = normalize_type_token(token)
        if normalized in self.type_nodes:
            return self.type_nodes[normalized]
        node = self.add_node(normalized, "type", {"full_text": token, "data_type": normalized})
        if node is not None:
            self.type_nodes[normalized] = node
        return node

    def as_symbolic(self, graph_kind: str, graph_features: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "graph_schema": GRAPH_SCHEMA,
            "graph_kind": graph_kind,
            "node_tokens": self.node_tokens,
            "node_types": self.node_types,
            "node_features": self.node_features,
            "edges": self.edges,
            "edge_features": self.edge_features,
            "graph_features": graph_features or {},
        }


def strip_comment(line: str) -> str:
    if ";" in line:
        line = line.split(";", 1)[0]
    return line.strip()


def instruction_opcode(text: str) -> str:
    rhs = text.split("=", 1)[1].strip() if "=" in text else text.strip()
    if re.search(r"\b(?:call|invoke)\b", rhs):
        callee_match = GLOBAL_RE.search(rhs)
        if callee_match:
            callee = normalize_token(callee_match.group(0).lstrip("@"))
            return normalize_token(f"call_{callee}")
    tokens = [normalize_token(tok) for tok in TOKEN_RE.findall(rhs)]
    for token in tokens:
        if token in LLVM_PREFIXES:
            continue
        if token.startswith("const_"):
            continue
        if is_llvm_type_token(token):
            continue
        return token
    return "unknown_op"


def parse_ir_blocks(ir: str) -> tuple[str | None, list[tuple[str, list[str]]]]:
    header: str | None = None
    blocks: list[tuple[str, list[str]]] = []
    current_name: str | None = None
    current_lines: list[str] = []
    in_body = False
    saw_block = False

    for raw_line in ir.splitlines():
        line = strip_comment(raw_line)
        if not line:
            continue
        if line.startswith("define "):
            header = line
            in_body = "{" in line
            continue
        if not in_body:
            continue
        if line == "}":
            break
        if line.endswith(":"):
            if current_name is not None:
                blocks.append((current_name, current_lines))
            current_name = line[:-1].strip().lstrip("%")
            current_lines = []
            saw_block = True
            continue
        if not saw_block:
            current_name = "entry"
            saw_block = True
        current_lines.append(line)

    if current_name is not None:
        blocks.append((current_name, current_lines))
    return header, blocks


def define_args_from_header(builder: MiniProgramlBuilder, header: str) -> None:
    header_value_types = typed_values(header)
    typed_args: list[tuple[int, str]] = []
    for arg in SSA_RE.findall(header):
        if arg in builder.defs:
            continue
        data_type = header_value_types.get(arg)
        node = builder.variable_node(arg, arg, {"origin": "argument"})
        if node is not None:
            builder.defs[arg] = node
            if data_type:
                typed_args.append((node, data_type))
    for node, data_type in typed_args:
        add_type_annotation(builder, node, data_type)


def add_type_annotation(
    builder: MiniProgramlBuilder,
    node: int | None,
    data_type: str | None,
    position: int = 0,
) -> None:
    if node is None or not data_type:
        return
    annotation = (node, data_type)
    if annotation in builder.type_annotations:
        return
    builder.type_annotations.add(annotation)
    type_node = builder.type_node(data_type)
    builder.add_edge(type_node, node, "type", position)


def program_line_value_type(line: str) -> str | None:
    rhs = line.split("=", 1)[1].strip() if "=" in line else line.strip()
    if rhs.startswith("ret "):
        return first_llvm_type(rhs)
    if rhs.startswith("store "):
        return first_llvm_type(rhs)
    if rhs.startswith(("br ", "switch ", "invoke ")):
        return None
    if re.match(r"^(?:icmp|fcmp)\b", rhs):
        return "i1"
    convert_match = re.search(r"\bto\s+(" + LLVM_TYPE_PATTERN + r")", rhs)
    if convert_match:
        return normalize_type_token(convert_match.group(1))
    if re.match(r"^alloca\b", rhs):
        return "ptr"
    return first_llvm_type(rhs)


def phi_incoming_blocks(line: str) -> set[str]:
    rhs = line.split("=", 1)[1] if "=" in line else line
    if not rhs.strip().startswith("phi "):
        return set()
    return {match.group("block") for match in PHI_INCOMING_RE.finditer(rhs)}


def add_program_instruction(
    builder: MiniProgramlBuilder,
    line: str,
    block_name: str,
    block_index: int,
    previous_inst: int | None,
) -> int | None:
    op = instruction_opcode(line)
    value_types = typed_values(line)
    line_data_type = program_line_value_type(line)
    inst_features: dict[str, Any] = {"full_text": line, "block": block_index, "block_name": block_name}
    inst = builder.add_node(op, "instruction", inst_features)
    if inst is None:
        return None
    if previous_inst is not None:
        builder.add_edge(previous_inst, inst, "control", 0)

    lhs = None
    if "=" in line:
        lhs_text = line.split("=", 1)[0].strip()
        match = SSA_RE.search(lhs_text)
        if match:
            lhs = match.group(0)
            var = builder.variable_node(lhs, lhs, {"origin": "ssa_def"})
            if var is not None:
                builder.defs[lhs] = var
                add_type_annotation(builder, var, line_data_type)
                builder.add_edge(inst, var, "data", 0, {"role": "def"})

    add_type_annotation(builder, inst, line_data_type)

    operand_position = 0
    phi_label_uses = {f"%{block}" for block in phi_incoming_blocks(line)}
    branch_label_uses = {f"%{block}" for block in LABEL_REF_RE.findall(line)}
    for use in SSA_RE.findall(line):
        if use == lhs or use in phi_label_uses or use in branch_label_uses:
            continue
        use_type = value_types.get(use, line_data_type)
        src = builder.defs.get(use)
        if src is not None and src >= 0:
            add_type_annotation(builder, src, use_type)
            builder.add_edge(src, inst, "data", operand_position, {"role": "use"})
        else:
            node = builder.variable_node(use, use, {"origin": "ssa_use"})
            add_type_annotation(builder, node, use_type)
            builder.add_edge(node, inst, "data", operand_position, {"role": "use"})
        operand_position += 1

    for global_name in sorted(set(GLOBAL_RE.findall(line))):
        node = builder.constant_node("global", global_name, {"global": True})
        builder.add_edge(node, inst, "data", operand_position, {"role": "global"})
        operand_position += 1

    for constant in ir_numeric_constants(line):
        constant_type = value_types.get(constant, line_data_type)
        node = builder.constant_node(normalize_token(constant), constant)
        add_type_annotation(builder, node, constant_type)
        builder.add_edge(node, inst, "data", operand_position, {"role": "constant"})
        operand_position += 1

    for match in LLVM_TYPE_RE.finditer(line):
        add_type_annotation(builder, inst, normalize_type_token(match.group(0)))
    return inst


def parse_program_ir_to_graph(
    ir: str,
    max_nodes: int = 512,
    max_edges: int = 4096,
    max_operand_nodes: int = 128,
) -> dict[str, Any]:
    builder = MiniProgramlBuilder(max_nodes=max_nodes, max_edges=max_edges, max_operand_nodes=max_operand_nodes)
    header, blocks = parse_ir_blocks(ir)
    if header:
        define_args_from_header(builder, header)

    block_order: dict[str, int] = {}
    block_first_inst: dict[str, int | None] = {}
    block_last_inst: dict[str, int | None] = {}
    block_phi_insts: dict[str, list[int]] = {}
    for index, (block_name, _lines) in enumerate(blocks):
        block_order[block_name] = index

    for block_name, lines in blocks:
        previous_inst: int | None = None
        first_inst: int | None = None
        for line in lines:
            previous_inst = add_program_instruction(builder, line, block_name, block_order[block_name], previous_inst)
            if first_inst is None:
                first_inst = previous_inst
            if previous_inst is not None and instruction_opcode(line) == "phi":
                block_phi_insts.setdefault(block_name, []).append(previous_inst)
        block_first_inst[block_name] = first_inst
        block_last_inst[block_name] = previous_inst

    for block_name, lines in blocks:
        if not lines:
            continue
        terminator = block_last_inst.get(block_name)
        if terminator is None:
            continue
        for successor_position, target in enumerate(LABEL_REF_RE.findall(lines[-1])):
            target_entries = block_phi_insts.get(target) or [block_first_inst.get(target)]
            if not target_entries:
                continue
            is_backedge = block_order.get(target, 10**9) <= block_order.get(block_name, -1)
            for target_entry in target_entries:
                if target_entry is None:
                    continue
                builder.add_edge(
                    terminator,
                    target_entry,
                    "control",
                    successor_position,
                    {
                        "backedge": is_backedge,
                        "phi_control": bool(block_phi_insts.get(target)),
                        "predecessor_block": block_name,
                        "successor_block": target,
                    },
                )

    return builder.as_symbolic(
        "program",
        {
            "source": "llvm_ir",
            "function_header": header,
            "blocks": [{"name": name, "index": block_order[name]} for name, _lines in blocks],
        },
    )


RESERVED_OPERATION_WORDS = {
    "and",
    "else",
    "endif",
    "endfor",
    "false",
    "fi",
    "for",
    "foreach",
    "if",
    "in",
    "is",
    "of",
    "or",
    "over",
    "then",
    "to",
    "true",
    "when",
    "while",
}

OPERATOR_WORDS = {
    "active_lanes",
    "add",
    "compare",
    "reduce_add",
    "sat",
    "saturate",
    "sext",
    "shift",
    "sign_extend",
    "signextend",
    "sub",
    "sum",
    "zero_extend",
    "zeroextend",
    "zext",
}


def operation_identifiers(text: str) -> list[str]:
    # Intel pseudocode frequently writes lane views as dst.i16[lane]. The suffix
    # is a type/view annotation; the graph variable is the base symbol dst.
    without_views = re.sub(r"\.[A-Za-z][A-Za-z0-9_]*", "", text)
    identifiers: list[str] = []
    seen: set[str] = set()
    for token in IDENT_RE.findall(without_views):
        normalized = normalize_token(token)
        if normalized in RESERVED_OPERATION_WORDS or normalized in OPERATOR_WORDS:
            continue
        if normalized not in seen:
            identifiers.append(normalized)
            seen.add(normalized)
    return identifiers


def bracket_identifiers(text: str) -> list[str]:
    identifiers: list[str] = []
    seen: set[str] = set()
    for bracket_text in re.findall(r"\[([^\]]+)\]", text):
        for identifier in operation_identifiers(bracket_text):
            if identifier not in seen:
                identifiers.append(identifier)
                seen.add(identifier)
    return identifiers


def numeric_constants(text: str) -> list[str]:
    constants: list[str] = []
    seen: set[str] = set()
    for token in NUMBER_RE.findall(text):
        if token not in seen:
            constants.append(token)
            seen.add(token)
    return constants


def ir_numeric_constants(text: str) -> list[str]:
    masked = SSA_RE.sub(" ", text)
    masked = GLOBAL_RE.sub(" ", masked)
    masked = LLVM_TYPE_RE.sub(" ", masked)
    constants: list[str] = []
    seen: set[str] = set()
    for match in NUMBER_RE.finditer(masked):
        start, end = match.span()
        before = masked[start - 1] if start > 0 else ""
        after = masked[end] if end < len(masked) else ""
        if before and (before.isalnum() or before in "_.%@"):
            continue
        if after and (after.isalnum() or after in "_."):
            continue
        token = match.group(0)
        if token not in seen:
            constants.append(token)
            seen.add(token)
    return constants


def first_identifier(text: str) -> str | None:
    identifiers = operation_identifiers(text)
    return identifiers[0] if identifiers else None


def expression_operators(text: str) -> list[str]:
    operators: list[str] = []
    for operator, pattern in OPERATOR_PATTERNS:
        if operator not in operators and pattern.search(text):
            operators.append(operator)
    return operators


def primary_expression_operator(text: str) -> str:
    lowered = text.lower()
    if "sum" in lowered or "reduce_add" in lowered:
        return "sum"
    if "+" in text or re.search(r"\badd\b", lowered):
        return "add"
    if "*" in text:
        return "mul"
    if "zero_extend" in lowered or "zeroextend" in lowered or "zext" in lowered:
        return "zero_extend"
    if "sign_extend" in lowered or "signextend" in lowered or "sext" in lowered:
        return "sign_extend"
    if "<<" in text or ">>" in text or "shift" in lowered:
        return "shift"
    if re.search(r"==|!=|<=|>=|<|>|\bcompare\b", text, re.IGNORECASE):
        return "compare"
    if "-" in text or re.search(r"\bsub\b", lowered):
        return "sub"
    return "assign"


def intrinsic_variable_node(
    builder: MiniProgramlBuilder,
    symbol_nodes: dict[str, int],
    name: str,
    features: dict[str, Any] | None = None,
) -> int | None:
    normalized = normalize_token(name)
    if not normalized or normalized == "empty":
        return None
    if normalized in symbol_nodes:
        node = symbol_nodes[normalized]
        merge_node_features(builder.node_features[node], features)
        return node
    node = builder.add_node(normalized, "variable", {"full_text": name, "symbol": normalized, **(features or {})})
    if node is not None:
        symbol_nodes[normalized] = node
    return node


def intrinsic_constant_node(
    builder: MiniProgramlBuilder,
    constant_nodes: dict[str, int],
    value: str,
) -> int | None:
    if value in constant_nodes:
        return constant_nodes[value]
    if builder.max_operand_nodes > 0 and builder.operand_nodes >= builder.max_operand_nodes:
        return None
    token = "const_" + value.replace("-", "neg").replace(".", "_")
    node = builder.add_node(token, "constant", {"full_text": value, "value": value})
    if node is not None:
        builder.operand_nodes += 1
        constant_nodes[value] = node
    return node


def add_intrinsic_operands(
    builder: MiniProgramlBuilder,
    record: dict[str, Any],
    symbol_nodes: dict[str, int],
) -> None:
    for operand in as_list(record.get("operands")):
        if not isinstance(operand, dict):
            continue
        name = normalize_token(str(operand.get("name", "")))
        if not name:
            continue
        operand_type = operand.get("type")
        features: dict[str, Any] = {"role": operand.get("role", "")}
        data_type = normalize_type_token(str(operand_type)) if operand_type else None
        operand_node = intrinsic_variable_node(builder, symbol_nodes, name, features)
        if operand_node is not None and operand_type:
            add_type_annotation(builder, operand_node, data_type)


def parse_loop_header(line: str) -> dict[str, Any] | None:
    match = FOR_ASSIGN_RE.search(line) or FOR_RANGE_RE.search(line)
    if match:
        start = int(match.group("start"))
        end = int(match.group("end"))
        trip_count = max(0, end - start + 1)
        return {
            "variable": normalize_token(match.group("var")),
            "start": start,
            "end": end,
            "inclusive": True,
            "iteration_count": trip_count,
            "backedge_count": max(0, trip_count - 1),
            "domain": f"{start}..{end}",
        }
    match = FOR_DOMAIN_RE.search(line)
    if match:
        return {
            "variable": normalize_token(match.group("var")),
            "domain": match.group("domain").strip(),
        }
    return None


def execution_features(scope_stack: list[dict[str, Any]]) -> dict[str, Any]:
    execution_count = 1
    has_static_loop = False
    for scope in scope_stack:
        if scope.get("kind") == "loop" and isinstance(scope.get("iteration_count"), int):
            execution_count *= int(scope["iteration_count"])
            has_static_loop = True
    if has_static_loop:
        return {"execution_count": execution_count}
    return {}


def innermost_loop(scope_stack: list[dict[str, Any]]) -> dict[str, Any] | None:
    for scope in reversed(scope_stack):
        if scope.get("kind") == "loop":
            return scope
    return None


def add_scoped_control_edge(
    builder: MiniProgramlBuilder,
    previous_control: int | None,
    node: int | None,
    scope_stack: list[dict[str, Any]],
    features: dict[str, Any] | None = None,
) -> None:
    if node is None:
        return
    edge_features = execution_features(scope_stack)
    edge_features.update(features or {})
    loop_scope = innermost_loop(scope_stack)
    if loop_scope is not None and previous_control == loop_scope.get("node"):
        edge_features.setdefault("scope", "loop_body")
    builder.add_edge(previous_control, node, "control", 0, edge_features)
    if loop_scope is not None and previous_control != loop_scope.get("node"):
        loop_edge_features = execution_features(scope_stack)
        loop_edge_features.update({"scope": "loop_body"})
        builder.add_edge(loop_scope.get("node"), node, "control", 1, loop_edge_features)


def add_data_edge(
    builder: MiniProgramlBuilder,
    src: int | None,
    dst: int | None,
    position: int,
    role: str,
    scope_stack: list[dict[str, Any]],
) -> None:
    features = execution_features(scope_stack)
    features["role"] = role
    builder.add_edge(src, dst, "data", position, features)


def add_constants_to_operation(
    builder: MiniProgramlBuilder,
    constant_nodes: dict[str, int],
    text: str,
    operation_node: int | None,
    start_position: int,
    scope_stack: list[dict[str, Any]],
) -> int:
    position = start_position
    for constant in numeric_constants(text):
        constant_node = intrinsic_constant_node(builder, constant_nodes, constant)
        add_data_edge(builder, constant_node, operation_node, position, "constant", scope_stack)
        position += 1
    return position


def add_identifier_uses(
    builder: MiniProgramlBuilder,
    symbol_nodes: dict[str, int],
    identifiers: list[str],
    operation_node: int | None,
    start_position: int,
    role: str,
    scope_stack: list[dict[str, Any]],
    skip: set[str] | None = None,
) -> int:
    position = start_position
    skipped = skip or set()
    for identifier in identifiers:
        if identifier in skipped:
            continue
        variable_node = intrinsic_variable_node(builder, symbol_nodes, identifier)
        add_data_edge(builder, variable_node, operation_node, position, role, scope_stack)
        position += 1
    return position


def ast_number_value(value: Any) -> int | float | None:
    if isinstance(value, Number):
        return value.val
    return None


def ast_operator_token(op: str) -> str:
    return {
        "+": "add",
        "-": "sub",
        "*": "mul",
        "/": "div",
        "%": "mod",
        "<<": "shift_left",
        ">>": "shift_right",
        "<<<": "logical_shift_left",
        ">>>": "logical_shift_right",
        "&": "and",
        "|": "or",
        "AND": "and",
        "OR": "or",
        "XOR": "xor",
        "==": "compare_eq",
        "!=": "compare_ne",
        "<": "compare_lt",
        ">": "compare_gt",
        "<=": "compare_le",
        ">=": "compare_ge",
    }.get(op, normalize_token(op))


def graph_edge_features(role: str, scope_stack: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    features = execution_features(scope_stack)
    features["role"] = role
    if extra:
        features.update(extra)
    return features


def add_ast_expr_graph(
    builder: MiniProgramlBuilder,
    symbol_nodes: dict[str, int],
    constant_nodes: dict[str, int],
    expr: Any,
    scope_stack: list[dict[str, Any]],
) -> int | None:
    if isinstance(expr, Var):
        return intrinsic_variable_node(builder, symbol_nodes, expr.name)
    if isinstance(expr, Number):
        return intrinsic_constant_node(builder, constant_nodes, str(expr.val))
    if isinstance(expr, Lookup):
        node = builder.add_node(
            "lookup",
            "instruction",
            {"full_text": f"{ast_expr_text(expr.obj)}.{expr.key}", "field": expr.key},
        )
        obj_node = add_ast_expr_graph(builder, symbol_nodes, constant_nodes, expr.obj, scope_stack)
        builder.add_edge(obj_node, node, "data", 0, graph_edge_features("base", scope_stack))
        return node
    if isinstance(expr, UnaryExpr):
        node = builder.add_node(
            "unary_" + ast_operator_token(expr.op),
            "instruction",
            {"full_text": ast_expr_text(expr), "operator": expr.op},
        )
        operand = add_ast_expr_graph(builder, symbol_nodes, constant_nodes, expr.a, scope_stack)
        builder.add_edge(operand, node, "data", 0, graph_edge_features("operand", scope_stack))
        return node
    if isinstance(expr, BinaryExpr):
        node = builder.add_node(
            ast_operator_token(expr.op),
            "instruction",
            {"full_text": ast_expr_text(expr), "operator": expr.op, "expr_id": expr.expr_id},
        )
        left = add_ast_expr_graph(builder, symbol_nodes, constant_nodes, expr.a, scope_stack)
        right = add_ast_expr_graph(builder, symbol_nodes, constant_nodes, expr.b, scope_stack)
        builder.add_edge(left, node, "data", 0, graph_edge_features("lhs", scope_stack))
        builder.add_edge(right, node, "data", 1, graph_edge_features("rhs", scope_stack))
        return node
    if isinstance(expr, BitSlice):
        node = builder.add_node(
            "slice",
            "instruction",
            {"full_text": ast_expr_text(expr), "slice": True},
        )
        base = add_ast_expr_graph(builder, symbol_nodes, constant_nodes, expr.bv, scope_stack)
        hi = add_ast_expr_graph(builder, symbol_nodes, constant_nodes, expr.hi, scope_stack)
        lo = add_ast_expr_graph(builder, symbol_nodes, constant_nodes, expr.lo, scope_stack)
        builder.add_edge(base, node, "data", 0, graph_edge_features("base", scope_stack))
        builder.add_edge(hi, node, "data", 1, graph_edge_features("hi", scope_stack))
        builder.add_edge(lo, node, "data", 2, graph_edge_features("lo", scope_stack))
        return node
    if isinstance(expr, Call):
        node = builder.add_node(
            normalize_token(expr.func),
            "instruction",
            {"full_text": ast_expr_text(expr), "call": expr.func},
        )
        for position, arg in enumerate(expr.args):
            arg_node = add_ast_expr_graph(builder, symbol_nodes, constant_nodes, arg, scope_stack)
            builder.add_edge(arg_node, node, "data", position, graph_edge_features("arg", scope_stack))
        return node
    node = builder.add_node("pseudo", "instruction", {"full_text": str(expr)})
    return node


def ast_expr_text(expr: Any) -> str:
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Number):
        return str(expr.val)
    if isinstance(expr, Lookup):
        return f"{ast_expr_text(expr.obj)}.{expr.key}"
    if isinstance(expr, UnaryExpr):
        return f"{expr.op}{ast_expr_text(expr.a)}"
    if isinstance(expr, BinaryExpr):
        return f"{ast_expr_text(expr.a)} {expr.op} {ast_expr_text(expr.b)}"
    if isinstance(expr, BitSlice):
        return f"{ast_expr_text(expr.bv)}[{ast_expr_text(expr.hi)}:{ast_expr_text(expr.lo)}]"
    if isinstance(expr, Call):
        return f"{expr.func}({', '.join(ast_expr_text(arg) for arg in expr.args)})"
    return str(expr)


def ast_lhs_base(expr: Any) -> str | None:
    if isinstance(expr, Var):
        return normalize_token(expr.name)
    if isinstance(expr, BitSlice):
        return ast_lhs_base(expr.bv)
    if isinstance(expr, Lookup):
        return ast_lhs_base(expr.obj)
    return None


def ensure_effect_node(
    builder: MiniProgramlBuilder,
    expr_node: int | None,
    fallback_token: str,
    full_text: str,
    scope_stack: list[dict[str, Any]],
) -> int | None:
    if expr_node is not None and builder.node_types[expr_node] == "instruction":
        builder.node_features[expr_node].setdefault("full_text", full_text)
        builder.node_features[expr_node].setdefault("effect", fallback_token)
        return expr_node
    node = builder.add_node(fallback_token, "instruction", {"full_text": full_text, "effect": fallback_token})
    builder.add_edge(expr_node, node, "data", 0, graph_edge_features("value", scope_stack))
    return node


def static_int(expr: Any) -> int | None:
    if isinstance(expr, Number) and isinstance(expr.val, int):
        return expr.val
    return None


def build_ast_stmt_graph(
    builder: MiniProgramlBuilder,
    symbol_nodes: dict[str, int],
    constant_nodes: dict[str, int],
    stmt: Any,
    previous_control: int | None,
    scope_stack: list[dict[str, Any]],
) -> int | None:
    if isinstance(stmt, For):
        begin_int = static_int(stmt.begin)
        end_int = static_int(stmt.end)
        iteration_count: int | None = None
        backedge_count: int | None = None
        if begin_int is not None and end_int is not None:
            distance = end_int - begin_int if stmt.inc else begin_int - end_int
            iteration_count = max(0, distance + 1)
            backedge_count = max(0, iteration_count - 1)
        loop_node = builder.add_node(
            "for",
            "instruction",
            {
                "full_text": f"FOR {stmt.iterator} := {ast_expr_text(stmt.begin)} {'TO' if stmt.inc else 'DOWNTO'} {ast_expr_text(stmt.end)}",
                "loop_variable": normalize_token(stmt.iterator),
                "iteration_count": iteration_count,
            },
        )
        control_features = {"loop_header": True}
        if iteration_count is not None:
            control_features["execution_count"] = iteration_count
        add_scoped_control_edge(builder, previous_control, loop_node, scope_stack, control_features)

        loop_var = intrinsic_variable_node(builder, symbol_nodes, stmt.iterator, {"role": "induction"})
        add_data_edge(builder, loop_node, loop_var, 0, "induction_def", scope_stack)
        begin_node = add_ast_expr_graph(builder, symbol_nodes, constant_nodes, stmt.begin, scope_stack)
        end_node = add_ast_expr_graph(builder, symbol_nodes, constant_nodes, stmt.end, scope_stack)
        add_data_edge(builder, begin_node, loop_node, 0, "lower_bound", scope_stack)
        add_data_edge(builder, end_node, loop_node, 1, "upper_bound", scope_stack)

        self_loop_features = {"backedge": True, "loop_self": True, "range_inclusive": True}
        if iteration_count is not None:
            self_loop_features.update(
                {
                    "iteration_count": iteration_count,
                    "backedge_count": backedge_count,
                    "execution_count": backedge_count,
                }
            )
            if begin_int is not None and end_int is not None:
                self_loop_features["range_start"] = begin_int
                self_loop_features["range_end"] = end_int
        builder.add_edge(loop_node, loop_node, "control", 0, self_loop_features)

        loop_scope = {
            "kind": "loop",
            "node": loop_node,
            "variable": normalize_token(stmt.iterator),
            "iteration_count": iteration_count,
        }
        body_previous: int | None = loop_node
        for child in stmt.body:
            body_previous = build_ast_stmt_graph(
                builder, symbol_nodes, constant_nodes, child, body_previous, [*scope_stack, loop_scope]
            )
        return loop_node

    if isinstance(stmt, If):
        predicate_value = add_ast_expr_graph(builder, symbol_nodes, constant_nodes, stmt.cond, scope_stack)
        predicate_node = ensure_effect_node(builder, predicate_value, "if", f"IF {ast_expr_text(stmt.cond)}", scope_stack)
        if predicate_node is not None:
            builder.node_features[predicate_node]["predicate"] = True
        add_scoped_control_edge(builder, previous_control, predicate_node, scope_stack, {"predicate": True})
        pred_scope = {"kind": "predicate", "node": predicate_node}
        branch_previous = predicate_node
        for child in stmt.then:
            branch_previous = build_ast_stmt_graph(
                builder, symbol_nodes, constant_nodes, child, branch_previous, [*scope_stack, pred_scope]
            )
        other_previous = predicate_node
        for child in stmt.otherwise:
            other_previous = build_ast_stmt_graph(
                builder, symbol_nodes, constant_nodes, child, other_previous, [*scope_stack, pred_scope]
            )
        return branch_previous or other_previous or predicate_node

    if isinstance(stmt, Update):
        rhs_node = add_ast_expr_graph(builder, symbol_nodes, constant_nodes, stmt.rhs, scope_stack)
        effect_node = ensure_effect_node(
            builder,
            rhs_node,
            "assign",
            f"{ast_expr_text(stmt.lhs)} := {ast_expr_text(stmt.rhs)}",
            scope_stack,
        )
        add_scoped_control_edge(builder, previous_control, effect_node, scope_stack)
        lhs_node = add_ast_expr_graph(builder, symbol_nodes, constant_nodes, stmt.lhs, scope_stack)
        if lhs_node is not None:
            builder.node_features[lhs_node].setdefault("role", "def")
        add_data_edge(builder, effect_node, lhs_node, 0, "def", scope_stack)
        base = ast_lhs_base(stmt.lhs)
        if base is not None:
            base_node = intrinsic_variable_node(builder, symbol_nodes, base)
            if base_node is not None:
                builder.node_features[base_node].setdefault("role", "def")
            add_data_edge(builder, effect_node, base_node, 1, "def_base", scope_stack)
        return effect_node

    if isinstance(stmt, PseudoStmt):
        node = builder.add_node("pseudo", "instruction", {"full_text": stmt.desc})
        add_scoped_control_edge(builder, previous_control, node, scope_stack)
        return node

    expr_node = add_ast_expr_graph(builder, symbol_nodes, constant_nodes, stmt, scope_stack)
    effect_node = ensure_effect_node(builder, expr_node, "effect", ast_expr_text(stmt), scope_stack)
    add_scoped_control_edge(builder, previous_control, effect_node, scope_stack)
    return effect_node


def parse_intrinsic_spec_to_graph(
    record: dict[str, Any],
    max_nodes: int = 0,
    max_edges: int = 0,
    max_operand_nodes: int = 0,
    root_token_mode: str = "generic",
) -> dict[str, Any]:
    builder = MiniProgramlBuilder(max_nodes=max_nodes, max_edges=max_edges, max_operand_nodes=max_operand_nodes)
    label = instruction_label(record)
    if root_token_mode == "label":
        root_token = label
    elif root_token_mode == "generic":
        root_token = "intrinsic_root"
    else:
        raise ValueError(f"unknown intrinsic root token mode: {root_token_mode}")
    graph_features = {
        "label": label,
        "architecture": first_text(record, "architecture", "isa"),
        "extension": first_text(record, "extension", "instruction_set"),
        "intrinsic": first_text(record, "llvm_intrinsic", "intrinsic"),
        "assembly": first_text(record, "instruction", "assembly"),
        "cpuid": as_list(record.get("cpuid")),
        "source": first_text(record, "source"),
    }
    root = builder.add_node(
        root_token,
        "instruction",
        {"full_text": first_text(record, "instruction", "assembly", "intrinsic", "label"), "semantic_root": True},
    )
    symbol_nodes: dict[str, int] = {}
    constant_nodes: dict[str, int] = {}
    add_intrinsic_operands(builder, record, symbol_nodes)

    previous_control: int | None = root
    ast = parse_operation_ast(record.get("operation", []))
    graph_features["operation_ast"] = ast_to_json(ast)
    for stmt in ast:
        previous_control = build_ast_stmt_graph(
            builder,
            symbol_nodes,
            constant_nodes,
            stmt,
            previous_control,
            [],
        )

    return builder.as_symbolic("intrinsic", graph_features)


def parse_mini_programl(
    mode: str,
    source: str | dict[str, Any],
    max_nodes: int = 512,
    max_edges: int = 4096,
    max_operand_nodes: int = 128,
    root_token_mode: str = "generic",
) -> dict[str, Any]:
    if mode == "program":
        return parse_program_ir_to_graph(str(source), max_nodes, max_edges, max_operand_nodes)
    if mode == "intrinsic":
        if not isinstance(source, dict):
            raise TypeError("intrinsic mode expects a spec record dictionary")
        return parse_intrinsic_spec_to_graph(source, max_nodes, max_edges, max_operand_nodes, root_token_mode)
    raise ValueError(f"unknown mini-programl parse mode: {mode}")
