#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

try:
    from label_iroptset_instructions import (
        DEFAULT_EXCLUDE_INTRINSIC_PREFIXES,
        extract_intrinsic_counts,
    )
    from mini_programl_graphs import (
        EDGE_FEATURE_TYPES,
        EDGE_TYPES,
        NODE_TYPES,
        build_vocab,
        compact_graph_features,
        compact_node_features,
        encode_graph,
        normalize_type_token,
        parse_program_ir_to_graph,
        write_jsonl,
    )
except ModuleNotFoundError:  # pragma: no cover - used when imported as a package module
    from scripts.label_iroptset_instructions import (
        DEFAULT_EXCLUDE_INTRINSIC_PREFIXES,
        extract_intrinsic_counts,
    )
    from scripts.mini_programl_graphs import (
        EDGE_FEATURE_TYPES,
        EDGE_TYPES,
        NODE_TYPES,
        build_vocab,
        compact_graph_features,
        compact_node_features,
        encode_graph,
        normalize_type_token,
        parse_program_ir_to_graph,
        write_jsonl,
    )


DEFINE_RE = re.compile(r"^\s*define\b.*@(?P<name>[A-Za-z0-9_.$-]+)\s*\(")
VECTOR_TYPE_RE = re.compile(
    r"<\s*(?P<lanes>\d+)\s+x\s+(?P<elem>ptr|i\d+|half|bfloat|float|double|x86_fp80|fp128|ppc_fp128)\s*>"
)
LLVM_COMMENT_RE = re.compile(r";.*$")

VECTOR_OPS = {
    "add",
    "fadd",
    "sub",
    "fsub",
    "mul",
    "fmul",
    "udiv",
    "sdiv",
    "fdiv",
    "urem",
    "srem",
    "and",
    "or",
    "xor",
    "shl",
    "lshr",
    "ashr",
    "icmp",
    "fcmp",
    "select",
    "load",
    "store",
    "sext",
    "zext",
    "trunc",
    "fpext",
    "fptrunc",
    "sitofp",
    "uitofp",
    "fptosi",
    "fptoui",
}

LABEL_TYPE_RE = re.compile(r"^(?:v\d+(?:i\d+|f\d+|p\d+)|i\d+|f\d+|p\d+)$")

LLVM_FLOAT_TYPES = {
    "f16": "half",
    "f32": "float",
    "f64": "double",
    "f80": "x86_fp80",
    "f128": "fp128",
}

BINARY_INTRINSIC_NAMES = {
    "copysign",
    "fshl",
    "fshr",
    "maxnum",
    "maximum",
    "minnum",
    "minimum",
    "pow",
    "powi",
}

TERNARY_INTRINSIC_NAMES = {"fmuladd"}

REDUCE_OP_SYMBOLS = {
    "add": "+",
    "fadd": "+",
    "mul": "*",
    "fmul": "*",
    "and": "&",
    "or": "|",
    "xor": "^",
}

SCALAR_CHANGE_OPS = {
    "add",
    "fadd",
    "sub",
    "fsub",
    "mul",
    "fmul",
    "udiv",
    "sdiv",
    "fdiv",
    "urem",
    "srem",
    "and",
    "or",
    "xor",
    "shl",
    "lshr",
    "ashr",
    "icmp",
    "fcmp",
    "select",
    "load",
    "store",
    "getelementptr",
    "sext",
    "zext",
    "trunc",
    "fpext",
    "fptrunc",
    "sitofp",
    "uitofp",
    "fptosi",
    "fptoui",
}

BINARY_VECTOR_OP_SYMBOLS = {
    "add": "+",
    "fadd": "+",
    "sub": "-",
    "fsub": "-",
    "mul": "*",
    "fmul": "*",
    "udiv": "/",
    "sdiv": "/",
    "fdiv": "/",
    "urem": "%",
    "srem": "%",
    "and": "AND",
    "or": "OR",
    "xor": "XOR",
    "shl": "<<",
    "lshr": ">>",
    "ashr": ">>",
}

DEFAULT_EXCLUDE_TARGET_LABEL_REGEXES = (
    r"^llvm\.vector\.(?:load|store)\.",
)


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def split_name_for_key(key: str) -> str:
    value = int(stable_hash(key)[:8], 16) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "val"
    return "test"


def strip_comment(line: str) -> str:
    return LLVM_COMMENT_RE.sub("", line).strip()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_defined_functions(ir: str) -> dict[str, str]:
    functions: dict[str, str] = {}
    lines = ir.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        match = DEFINE_RE.match(line)
        if not match:
            index += 1
            continue
        name = match.group("name")
        body_lines = [line]
        brace_depth = line.count("{") - line.count("}")
        index += 1
        while index < len(lines):
            body_lines.append(lines[index])
            brace_depth += lines[index].count("{") - lines[index].count("}")
            index += 1
            if brace_depth <= 0 and body_lines[-1].strip().endswith("}"):
                break
        functions[name] = "\n".join(body_lines) + "\n"
    return functions


def run_opt(opt: str, input_ir: str, passes: str, extra_args: list[str], timeout: int) -> tuple[bool, str, str]:
    command = [opt, "-S", f"-passes={passes}", *extra_args]
    proc = subprocess.run(
        command,
        input=input_ir,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode == 0, proc.stdout, proc.stderr


def suite_for_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "loopvectorize" in parts:
        return "loop-vectorize-x86"
    if "slpvectorizer" in parts:
        return "slp-vectorizer-x86"
    return "unknown"


def passes_for_path(path: Path, loop_passes: str, slp_passes: str, default_passes: str) -> str:
    suite = suite_for_path(path)
    if suite == "loop-vectorize-x86":
        return loop_passes
    if suite == "slp-vectorizer-x86":
        return slp_passes
    return default_passes


def discover_test_files(args: argparse.Namespace) -> list[Path]:
    dirs: list[Path] = []
    if args.llvm_project_root:
        root = args.llvm_project_root
        dirs.extend(
            [
                root / "llvm" / "test" / "Transforms" / "LoopVectorize" / "X86",
                root / "llvm" / "test" / "Transforms" / "SLPVectorizer" / "X86",
            ]
        )
    dirs.extend(args.test_dir or [])
    files: list[Path] = []
    for directory in dirs:
        if not directory.exists():
            continue
        files.extend(sorted(directory.rglob("*.ll")))
    if args.ll_file:
        files.extend(args.ll_file)
    unique: dict[str, Path] = {}
    for path in files:
        unique[str(path.resolve())] = path
    return sorted(unique.values(), key=lambda item: str(item))


def instruction_opcode_from_line(text: str) -> str:
    line = strip_comment(text)
    rhs = line.split("=", 1)[1].strip() if "=" in line else line
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.-]*", rhs)
    for token in tokens:
        lowered = token.lower()
        if lowered in {"tail", "fast", "nuw", "nsw", "exact", "inbounds", "volatile", "atomic"}:
            continue
        if lowered in {"void", "ptr", "label", "metadata", "float", "double"} or re.fullmatch(r"i\d+", lowered):
            continue
        return lowered
    return "unknown"


def instruction_signature(text: str) -> str:
    line = strip_comment(text).lower()
    line = re.sub(r"%[A-Za-z0-9_.$-]+", "%v", line)
    line = re.sub(r"@[A-Za-z0-9_.$-]+", "@g", line)
    line = re.sub(r"<\s*\d+\s+x\s+[^>]+>", "<vec>", line)
    line = re.sub(r"\b\d+\b", "#", line)
    line = re.sub(r"\s+", " ", line)
    return line


def vector_type_in_line(line: str) -> str | None:
    match = VECTOR_TYPE_RE.search(line)
    if not match:
        return None
    return normalize_type_token(match.group(0))


def extract_vector_operation_counts(ir: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for raw_line in ir.splitlines():
        line = strip_comment(raw_line)
        if not line or line.endswith(":") or line.startswith(("define ", "declare ", "attributes ")):
            continue
        data_type = vector_type_in_line(line)
        if not data_type:
            continue
        opcode = instruction_opcode_from_line(line)
        if opcode not in VECTOR_OPS:
            continue
        counts[f"llvm.vector.{opcode}.{data_type}"] += 1
    return counts


def collect_after_labels(
    after_ir: str,
    label_source: str,
    include_intrinsic_prefixes: tuple[str, ...],
    exclude_intrinsic_prefixes: tuple[str, ...],
) -> Counter[str]:
    labels: Counter[str] = Counter()
    if label_source in {"intrinsic", "both"}:
        labels.update(extract_intrinsic_counts(after_ir, include_intrinsic_prefixes, exclude_intrinsic_prefixes))
    if label_source in {"vector-op", "both"}:
        labels.update(extract_vector_operation_counts(after_ir))
    return labels


def compile_label_regexes(patterns: list[str] | tuple[str, ...]) -> list[re.Pattern[str]]:
    return [re.compile(pattern) for pattern in patterns if pattern]


def filter_label_counter(labels: Counter[str], exclude_regexes: list[re.Pattern[str]]) -> Counter[str]:
    if not exclude_regexes:
        return Counter(labels)
    filtered: Counter[str] = Counter()
    for label, count in labels.items():
        if any(pattern.search(label) for pattern in exclude_regexes):
            continue
        filtered[label] = count
    return filtered


def graph_instruction_indices(graph: dict[str, Any]) -> list[int]:
    return [index for index, node_type in enumerate(graph["node_types"]) if node_type == "instruction"]


def changed_instruction_nodes(before_graph: dict[str, Any], after_ir: str) -> set[int]:
    after_counts = Counter(
        instruction_signature(line)
        for line in after_ir.splitlines()
        if strip_comment(line) and not strip_comment(line).endswith(":")
    )
    changed: set[int] = set()
    for node in graph_instruction_indices(before_graph):
        text = str(before_graph["node_features"][node].get("full_text", ""))
        signature = instruction_signature(text)
        if after_counts[signature] > 0:
            after_counts[signature] -= 1
            continue
        changed.add(node)

    if changed:
        return changed

    for node in graph_instruction_indices(before_graph):
        token = str(before_graph["node_tokens"][node])
        if token in SCALAR_CHANGE_OPS:
            changed.add(node)
    return changed


def graph_adjacency(graph: dict[str, Any], edge_types: set[str] | None = None) -> list[set[int]]:
    adjacency: list[set[int]] = [set() for _ in graph["node_tokens"]]
    for src, dst, edge_type, _position in graph["edges"]:
        if edge_types is not None and edge_type not in edge_types:
            continue
        adjacency[int(src)].add(int(dst))
        adjacency[int(dst)].add(int(src))
    return adjacency


def add_adjacent_data_and_type_nodes(graph: dict[str, Any], selected: set[int]) -> set[int]:
    expanded = set(selected)
    for src, dst, edge_type, _position in graph["edges"]:
        src = int(src)
        dst = int(dst)
        if edge_type in {"data", "type"} and (src in selected or dst in selected):
            expanded.add(src)
            expanded.add(dst)
    return expanded


def subgraph_symbolic(graph: dict[str, Any], selected_nodes: set[int]) -> dict[str, Any]:
    ordered = sorted(selected_nodes)
    remap = {old: new for new, old in enumerate(ordered)}
    edges: list[tuple[int, int, str, int]] = []
    edge_features: list[dict[str, Any]] = []
    for edge_index, (src, dst, edge_type, position) in enumerate(graph["edges"]):
        src = int(src)
        dst = int(dst)
        if src not in remap or dst not in remap:
            continue
        edges.append((remap[src], remap[dst], edge_type, int(position)))
        edge_features.append(dict(graph["edge_features"][edge_index]))
    return {
        "graph_schema": graph.get("graph_schema", "mini-programl"),
        "graph_kind": "changed_region",
        "node_tokens": [graph["node_tokens"][node] for node in ordered],
        "node_types": [graph["node_types"][node] for node in ordered],
        "node_features": [dict(graph["node_features"][node]) for node in ordered],
        "edges": edges,
        "edge_features": edge_features,
        "graph_features": {
            "source_graph_kind": graph.get("graph_kind", "program"),
            "source_nodes": ordered,
        },
    }


def producer_consumer_candidates(graph: dict[str, Any]) -> list[set[int]]:
    defs_by_value: dict[int, list[int]] = defaultdict(list)
    uses_by_value: dict[int, list[int]] = defaultdict(list)
    for edge_index, (src, dst, edge_type, _position) in enumerate(graph["edges"]):
        if edge_type != "data":
            continue
        features = graph["edge_features"][edge_index]
        role = features.get("role")
        src = int(src)
        dst = int(dst)
        if role == "def" and graph["node_types"][src] == "instruction":
            defs_by_value[dst].append(src)
        elif role in {"use", "operand", "lhs", "rhs", "value"} and graph["node_types"][dst] == "instruction":
            uses_by_value[src].append(dst)

    candidates: list[set[int]] = []
    for value_node, producers in defs_by_value.items():
        for producer in producers:
            for consumer in uses_by_value.get(value_node, []):
                selected = {producer, value_node, consumer}
                candidates.append(add_adjacent_data_and_type_nodes(graph, selected))
    return candidates


def store_backward_candidates(
    graph: dict[str, Any],
    max_depth: int = 5,
    max_nodes: int = 128,
) -> list[set[int]]:
    defs_by_value: dict[int, list[int]] = defaultdict(list)
    incoming_data_by_instruction: dict[int, list[int]] = defaultdict(list)
    for edge_index, (src, dst, edge_type, _position) in enumerate(graph["edges"]):
        if edge_type != "data":
            continue
        features = graph["edge_features"][edge_index]
        role = features.get("role")
        src = int(src)
        dst = int(dst)
        if role == "def" and graph["node_types"][src] == "instruction":
            defs_by_value[dst].append(src)
        elif graph["node_types"][dst] == "instruction":
            incoming_data_by_instruction[dst].append(src)

    store_nodes = [
        index
        for index, token in enumerate(graph["node_tokens"])
        if graph["node_types"][index] == "instruction" and token == "store"
    ]
    candidates: list[set[int]] = []
    for store in store_nodes:
        selected: set[int] = {store}
        frontier: list[tuple[int, int]] = [(store, 0)]
        seen_instructions: set[int] = {store}
        while frontier:
            instruction, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for operand in incoming_data_by_instruction.get(instruction, []):
                selected.add(operand)
                if max_nodes > 0 and len(selected) > max_nodes:
                    break
                for producer in defs_by_value.get(operand, []):
                    selected.add(producer)
                    if producer not in seen_instructions:
                        seen_instructions.add(producer)
                        frontier.append((producer, depth + 1))
                    if max_nodes > 0 and len(selected) > max_nodes:
                        break
                if max_nodes > 0 and len(selected) > max_nodes:
                    break
            if max_nodes > 0 and len(selected) > max_nodes:
                break
        expanded = add_adjacent_data_and_type_nodes(graph, selected)
        if max_nodes > 0 and len(expanded) > max_nodes:
            continue
        candidates.append(expanded)
    return candidates


def loop_backedge_candidates(graph: dict[str, Any]) -> list[set[int]]:
    candidates: list[set[int]] = []
    for edge_index, (src, dst, edge_type, _position) in enumerate(graph["edges"]):
        if edge_type != "control" or not graph["edge_features"][edge_index].get("backedge"):
            continue
        src_block = graph["node_features"][int(src)].get("block")
        dst_block = graph["node_features"][int(dst)].get("block")
        if not isinstance(src_block, int) or not isinstance(dst_block, int):
            continue
        lo, hi = sorted((src_block, dst_block))
        selected = {
            node
            for node, features in enumerate(graph["node_features"])
            if graph["node_types"][node] == "instruction"
            and isinstance(features.get("block"), int)
            and lo <= int(features["block"]) <= hi
        }
        if selected:
            candidates.append(add_adjacent_data_and_type_nodes(graph, selected))
    return candidates


def parse_candidate_modes(text: str) -> tuple[str, ...]:
    aliases = {
        "producer-consumer": "producer_consumer",
        "producer_consumer": "producer_consumer",
        "store-backward": "store_backward",
        "store_backward": "store_backward",
        "loop-backedge": "loop_backedge",
        "loop_backedge": "loop_backedge",
    }
    modes: list[str] = []
    for item in text.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"unknown candidate mode: {item}")
        modes.append(aliases[key])
    return tuple(dict.fromkeys(modes))


def candidate_regions(
    graph: dict[str, Any],
    modes: tuple[str, ...] | None = None,
    store_slice_depth: int = 5,
    store_slice_max_nodes: int = 128,
) -> list[set[int]]:
    selected_modes = modes or ("store_backward", "loop_backedge")
    candidates: list[set[int]] = []
    if "store_backward" in selected_modes:
        candidates.extend(
            store_backward_candidates(
                graph,
                max_depth=store_slice_depth,
                max_nodes=store_slice_max_nodes,
            )
        )
    if "producer_consumer" in selected_modes:
        candidates.extend(producer_consumer_candidates(graph))
    if "loop_backedge" in selected_modes:
        candidates.extend(loop_backedge_candidates(graph))
    deduped: list[set[int]] = []
    seen: set[tuple[int, ...]] = set()
    for selected in candidates:
        key = tuple(sorted(selected))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(set(key))
    return deduped


def region_overlap(selected: set[int], changed_nodes: set[int]) -> tuple[int, float]:
    instruction_nodes = selected & changed_nodes
    changed_count = len(instruction_nodes)
    selected_instruction_count = max(1, len(selected))
    return changed_count, changed_count / selected_instruction_count


def split_prefixes(values: list[str] | None) -> tuple[str, ...]:
    prefixes: list[str] = []
    for value in values or []:
        for part in value.split(","):
            part = part.strip().lower()
            if part:
                prefixes.append(part)
    return tuple(dict.fromkeys(prefixes))


def vector_label_spec(label: str) -> dict[str, Any] | None:
    match = re.fullmatch(r"llvm\.vector\.([a-z0-9_]+)\.(v\d+x.+)", label)
    if not match:
        return None
    opcode = match.group(1)
    data_type = match.group(2)
    lane_match = re.match(r"v(\d+)x(.+)", data_type)
    if not lane_match:
        return None
    lanes = int(lane_match.group(1))
    end = max(0, lanes - 1)
    operands = [{"name": "dst", "role": "dst", "type": data_type}]
    operation: list[str]
    if opcode in BINARY_VECTOR_OP_SYMBOLS:
        operands.extend(
            [
                {"name": "a", "role": "src", "type": data_type},
                {"name": "b", "role": "src", "type": data_type},
            ]
        )
        symbol = BINARY_VECTOR_OP_SYMBOLS[opcode]
        operation = [
            f"FOR lane := 0 to {end}",
            f"dst[lane] := a[lane] {symbol} b[lane]",
            "ENDFOR",
        ]
    elif opcode in {"icmp", "fcmp"}:
        operands.extend(
            [
                {"name": "a", "role": "src", "type": data_type},
                {"name": "b", "role": "src", "type": data_type},
            ]
        )
        operation = [
            f"FOR lane := 0 to {end}",
            "dst[lane] := compare(a[lane], b[lane])",
            "ENDFOR",
        ]
    elif opcode == "select":
        operands.extend(
            [
                {"name": "mask", "role": "predicate", "type": data_type},
                {"name": "a", "role": "src", "type": data_type},
                {"name": "b", "role": "src", "type": data_type},
            ]
        )
        operation = [
            f"FOR lane := 0 to {end}",
            "IF mask[lane] == 1",
            "dst[lane] := a[lane]",
            "ELSE",
            "dst[lane] := b[lane]",
            "ENDIF",
            "ENDFOR",
        ]
    elif opcode == "load":
        operands.append({"name": "mem", "role": "src", "type": data_type})
        operation = [f"FOR lane := 0 to {end}", "dst[lane] := mem[lane]", "ENDFOR"]
    elif opcode == "store":
        operands = [
            {"name": "mem", "role": "dst", "type": data_type},
            {"name": "value", "role": "src", "type": data_type},
        ]
        operation = [f"FOR lane := 0 to {end}", "mem[lane] := value[lane]", "ENDFOR"]
    else:
        operands.append({"name": "a", "role": "src", "type": data_type})
        operation = [f"FOR lane := 0 to {end}", f"dst[lane] := {opcode}(a[lane])", "ENDFOR"]
    return {
        "label": label,
        "architecture": "llvm-ir",
        "extension": "vector-ir",
        "intrinsic": label,
        "instruction": f"{opcode} {data_type}",
        "cpuid": [],
        "operands": operands,
        "operation": operation,
        "split": "train",
        "source": "synthetic-vector-ir",
    }


def llvm_label_type_token(token: str) -> str:
    vector_match = re.fullmatch(r"v(\d+)(i\d+|f\d+|p\d+)", token)
    if vector_match:
        return f"v{vector_match.group(1)}x{llvm_label_type_token(vector_match.group(2))}"
    if token in LLVM_FLOAT_TYPES:
        return LLVM_FLOAT_TYPES[token]
    if re.fullmatch(r"i\d+", token):
        return token
    if re.fullmatch(r"p\d+", token):
        return "ptr"
    return normalize_type_token(token)


def llvm_label_type_parts(label: str) -> list[str]:
    return [part for part in label.split(".") if LABEL_TYPE_RE.fullmatch(part)]


def scalar_type_from_vector_type(data_type: str) -> str:
    match = re.fullmatch(r"v\d+x(.+)", data_type)
    return match.group(1) if match else data_type


def llvm_label_base_name(label: str) -> str:
    parts = label.split(".")[1:]
    while parts and LABEL_TYPE_RE.fullmatch(parts[-1]):
        parts.pop()
    return ".".join(parts)


def operation_function_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "llvm_intrinsic"


def llvm_intrinsic_arity(base_name: str) -> int:
    if "with.overflow" in base_name:
        return 2
    tail = base_name.split(".")[-1]
    if tail in TERNARY_INTRINSIC_NAMES:
        return 3
    if tail in BINARY_INTRINSIC_NAMES:
        return 2
    if base_name.startswith("masked.store"):
        return 3
    if base_name.startswith("masked.load"):
        return 2
    if base_name.startswith("masked.scatter"):
        return 3
    if base_name.startswith("masked.gather"):
        return 2
    return 1


def reduction_label_spec(label: str) -> dict[str, Any] | None:
    match = re.fullmatch(r"llvm\.vector\.reduce\.([a-z0-9_]+)\.(v\d+(?:i\d+|f\d+))", label)
    if not match:
        return None
    opcode = match.group(1)
    data_type = llvm_label_type_token(match.group(2))
    lane_match = re.fullmatch(r"v(\d+)x(.+)", data_type)
    if not lane_match:
        return None
    lanes = int(lane_match.group(1))
    scalar_type = lane_match.group(2)
    end = max(0, lanes - 1)
    reduce_name = operation_function_name(f"reduce_{opcode}")
    operands = [
        {"name": "dst", "role": "dst", "type": scalar_type},
        {"name": "a", "role": "src", "type": data_type},
    ]
    if opcode in REDUCE_OP_SYMBOLS:
        symbol = REDUCE_OP_SYMBOLS[opcode]
        operation = [
            "acc := a[0]",
            f"FOR lane := 1 to {end}",
            f"acc := acc {symbol} a[lane]",
            "ENDFOR",
            "dst := acc",
        ]
    else:
        operation = [
            "acc := a[0]",
            f"FOR lane := 1 to {end}",
            f"acc := {reduce_name}(acc, a[lane])",
            "ENDFOR",
            "dst := acc",
        ]
    return {
        "label": label,
        "architecture": "llvm-ir",
        "extension": "llvm-intrinsic",
        "intrinsic": label,
        "instruction": label.removeprefix("llvm."),
        "cpuid": [],
        "operands": operands,
        "operation": operation,
        "split": "train",
        "source": "synthetic-llvm-intrinsic",
    }


def fallback_llvm_intrinsic_spec(label: str) -> dict[str, Any] | None:
    if not label.startswith("llvm."):
        return None
    if spec := vector_label_spec(label):
        return spec
    if spec := reduction_label_spec(label):
        return spec
    type_parts = llvm_label_type_parts(label)
    if not type_parts:
        return None
    data_type = llvm_label_type_token(next((part for part in type_parts if part.startswith("v")), type_parts[-1]))
    base_name = llvm_label_base_name(label)
    func_name = operation_function_name(base_name)
    arity = llvm_intrinsic_arity(base_name)
    source_type = data_type
    indexed = re.fullmatch(r"v(\d+)x(.+)", data_type)
    operands = [{"name": "dst", "role": "dst", "type": data_type}]
    operands.extend({"name": chr(ord("a") + index), "role": "src", "type": source_type} for index in range(arity))
    args = [chr(ord("a") + index) for index in range(arity)]
    if indexed:
        lanes = int(indexed.group(1))
        end = max(0, lanes - 1)
        indexed_args = ", ".join(f"{arg}[lane]" for arg in args)
        operation = [
            f"FOR lane := 0 to {end}",
            f"dst[lane] := {func_name}({indexed_args})",
            "ENDFOR",
        ]
    else:
        operation = [f"dst := {func_name}({', '.join(args)})"]
    return {
        "label": label,
        "architecture": "llvm-ir",
        "extension": "llvm-intrinsic",
        "intrinsic": label,
        "instruction": label.removeprefix("llvm."),
        "cpuid": [],
        "operands": operands,
        "operation": operation,
        "split": "train",
        "source": "synthetic-llvm-intrinsic",
    }


def make_region_record(
    symbolic: dict[str, Any],
    token_to_id: dict[str, int],
    metadata: dict[str, Any],
    include_symbols: bool,
    include_debug_text: bool,
) -> dict[str, Any]:
    encoded = encode_graph(symbolic, token_to_id)
    record = {
        **metadata,
        "graph_schema": symbolic.get("graph_schema", "mini-programl"),
        "graph_kind": symbolic.get("graph_kind", "changed_region"),
        "num_nodes": len(symbolic["node_tokens"]),
        "num_edges": len(symbolic["edges"]),
        "node_token_ids": encoded["node_token_ids"],
        "node_type_ids": encoded["node_type_ids"],
        "edges": encoded["edges"],
        "edge_feature_ids": encoded["edge_feature_ids"],
        "edge_counts": encoded["edge_counts"],
    }
    if include_symbols:
        record["node_tokens"] = symbolic["node_tokens"]
        record["node_types"] = symbolic["node_types"]
        record["node_features"] = compact_node_features(symbolic["node_features"], include_debug_text)
        record["symbolic_edges"] = symbolic["edges"]
        record["edge_features"] = symbolic["edge_features"]
        record["graph_features"] = compact_graph_features(symbolic.get("graph_features", {}), include_debug_text)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build diff-supervised changed-region graphs from LLVM vectorizer tests."
    )
    parser.add_argument("--llvm-project-root", type=Path)
    parser.add_argument("--test-dir", type=Path, action="append")
    parser.add_argument("--ll-file", type=Path, action="append")
    parser.add_argument("--opt", default="opt")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/llvm_vectorize_regions/x86_v1"))
    parser.add_argument("--loop-passes", default="loop-vectorize")
    parser.add_argument("--slp-passes", default="slp-vectorizer")
    parser.add_argument("--default-passes", default="loop-vectorize")
    parser.add_argument("--opt-extra-arg", action="append", default=[])
    parser.add_argument("--opt-timeout", type=int, default=30)
    parser.add_argument("--label-source", choices=["intrinsic", "vector-op", "both"], default="both")
    parser.add_argument("--include-intrinsic-prefix", action="append")
    parser.add_argument("--exclude-intrinsic-prefix", action="append")
    parser.add_argument(
        "--keep-memory-target-labels",
        action="store_true",
        help="Keep generic llvm.vector.load/store labels as supervised targets.",
    )
    parser.add_argument(
        "--exclude-target-label-regex",
        action="append",
        default=[],
        help="Additional regex for labels to remove from supervised target labels.",
    )
    parser.add_argument("--max-token-vocab", type=int, default=20000)
    parser.add_argument("--min-token-count", type=int, default=1)
    parser.add_argument("--min-positive-overlap", type=int, default=1)
    parser.add_argument("--max-negatives-per-function", type=int, default=8)
    parser.add_argument(
        "--candidate-modes",
        default="store-backward,loop-backedge",
        help="Comma-separated region proposal modes: store-backward, loop-backedge, producer-consumer.",
    )
    parser.add_argument("--store-slice-depth", type=int, default=5)
    parser.add_argument("--store-slice-max-nodes", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--drop-symbols", action="store_true")
    parser.add_argument("--include-debug-text", action="store_true")
    args = parser.parse_args()

    test_files = discover_test_files(args)
    if args.limit > 0:
        test_files = test_files[: args.limit]
    if not test_files:
        raise SystemExit("no .ll tests found; pass --llvm-project-root, --test-dir, or --ll-file")

    include_intrinsic_prefixes = split_prefixes(args.include_intrinsic_prefix)
    exclude_intrinsic_prefixes = tuple(
        dict.fromkeys((*DEFAULT_EXCLUDE_INTRINSIC_PREFIXES, *split_prefixes(args.exclude_intrinsic_prefix)))
    )
    try:
        candidate_modes = parse_candidate_modes(args.candidate_modes)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    exclude_target_patterns = [
        *(() if args.keep_memory_target_labels else DEFAULT_EXCLUDE_TARGET_LABEL_REGEXES),
        *args.exclude_target_label_regex,
    ]
    exclude_target_regexes = compile_label_regexes(exclude_target_patterns)

    symbolic_rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    failures: list[dict[str, Any]] = []
    label_counter: Counter[str] = Counter()
    vector_spec_labels: set[str] = set()
    stats: Counter[str] = Counter()

    for path in test_files:
        before_ir = read_text(path)
        passes = passes_for_path(path, args.loop_passes, args.slp_passes, args.default_passes)
        ok, after_ir, stderr = run_opt(args.opt, before_ir, passes, args.opt_extra_arg, args.opt_timeout)
        if not ok:
            failures.append({"file": str(path), "passes": passes, "error": stderr.strip()[-4000:]})
            stats["opt_failed"] += 1
            continue

        before_functions = extract_defined_functions(before_ir)
        after_functions = extract_defined_functions(after_ir)
        common_names = sorted(set(before_functions) & set(after_functions))
        if not common_names:
            stats["no_common_functions"] += 1
            continue

        for function_name in common_names:
            before_fn = before_functions[function_name]
            after_fn = after_functions[function_name]
            raw_labels = collect_after_labels(
                after_fn,
                args.label_source,
                include_intrinsic_prefixes,
                exclude_intrinsic_prefixes,
            )
            if not raw_labels:
                stats["no_after_labels"] += 1
                continue
            labels = filter_label_counter(raw_labels, exclude_target_regexes)
            stats["raw_after_label_occurrences"] += sum(raw_labels.values())
            stats["target_label_occurrences"] += sum(labels.values())
            stats["excluded_target_label_occurrences"] += sum(raw_labels.values()) - sum(labels.values())
            try:
                before_graph = parse_program_ir_to_graph(before_fn)
            except Exception as exc:  # pragma: no cover - defensive for unusual LLVM IR tests
                failures.append({"file": str(path), "function": function_name, "error": str(exc)})
                stats["parse_failed"] += 1
                continue

            changed_nodes = changed_instruction_nodes(before_graph, after_fn)
            candidates = candidate_regions(
                before_graph,
                modes=candidate_modes,
                store_slice_depth=args.store_slice_depth,
                store_slice_max_nodes=args.store_slice_max_nodes,
            )
            if not candidates:
                stats["no_candidates"] += 1
                continue

            positives = 0
            negatives = 0
            split = split_name_for_key(f"{path}:{function_name}")
            for candidate_index, selected in enumerate(candidates):
                overlap_count, _overlap_ratio = region_overlap(selected, changed_nodes)
                change_label = 1 if overlap_count >= args.min_positive_overlap else 0
                if change_label:
                    positives += 1
                elif negatives >= args.max_negatives_per_function:
                    continue
                else:
                    negatives += 1
                symbolic = subgraph_symbolic(before_graph, selected)
                row_id = stable_hash(f"{path}:{function_name}:{candidate_index}:{sorted(selected)}")[:16]
                target_labels = sorted(labels) if change_label else []
                metadata = {
                    "region_id": row_id,
                    "split": split,
                    "source_file": str(path),
                    "suite": suite_for_path(path),
                    "function_name": function_name,
                    "passes": passes,
                    "change_label": change_label,
                    "target_labels": target_labels,
                    "target_label_counts": {label: int(labels[label]) for label in target_labels},
                    "raw_target_label_counts": {label: int(raw_labels[label]) for label in sorted(raw_labels)},
                    "changed_instruction_overlap": overlap_count,
                    "changed_instruction_total": len(changed_nodes),
                    "source_node_count": len(selected),
                    "before_hash": stable_hash(before_fn),
                    "after_hash": stable_hash(after_fn),
                }
                symbolic_rows.append((split, symbolic, metadata))
                label_counter.update(target_labels)
                vector_spec_labels.update(label for label in target_labels if label.startswith("llvm."))

            stats["functions_with_regions"] += 1
            stats["positive_regions"] += positives
            stats["negative_regions"] += negatives

    if not symbolic_rows:
        raise SystemExit("no region rows built; inspect opt failures and label extraction settings")

    token_counter: Counter[str] = Counter()
    for split, symbolic, _metadata in symbolic_rows:
        if split == "train":
            token_counter.update(symbolic["node_tokens"])
    if not token_counter:
        for _split, symbolic, _metadata in symbolic_rows:
            token_counter.update(symbolic["node_tokens"])
    token_to_id = build_vocab(token_counter, args.max_token_vocab, args.min_token_count)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split, symbolic, metadata in symbolic_rows:
        rows_by_split[split].append(
            make_region_record(
                symbolic,
                token_to_id,
                metadata,
                include_symbols=not args.drop_symbols,
                include_debug_text=args.include_debug_text,
            )
        )

    for split in ("train", "val", "test"):
        rows = rows_by_split.get(split, [])
        write_jsonl(args.out_dir / f"{split}.regions.jsonl", rows)

    vector_specs = [spec for label in sorted(vector_spec_labels) if (spec := fallback_llvm_intrinsic_spec(label)) is not None]
    if vector_specs:
        write_jsonl(args.out_dir / "vector_instruction_specs.jsonl", vector_specs)

    vocab = {
        "token_to_id": token_to_id,
        "node_type_to_id": NODE_TYPES,
        "edge_type_to_id": EDGE_TYPES,
        "edge_feature_to_id": EDGE_FEATURE_TYPES,
        "label_to_id": {label: index for index, label in enumerate(sorted(label_counter))},
        "params": {
            "graph_schema": "mini-programl",
            "graph_kind": "changed_region",
            "label_source": args.label_source,
            "exclude_target_label_regexes": exclude_target_patterns,
            "candidate_modes": list(candidate_modes),
            "store_slice_depth": args.store_slice_depth,
            "store_slice_max_nodes": args.store_slice_max_nodes,
            "min_positive_overlap": args.min_positive_overlap,
            "max_negatives_per_function": args.max_negatives_per_function,
        },
    }
    (args.out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    summary = {
        "input_files": len(test_files),
        "rows": {split: len(rows_by_split.get(split, [])) for split in ("train", "val", "test")},
        "stats": dict(stats),
        "labels": label_counter.most_common(),
        "vector_instruction_specs": str(args.out_dir / "vector_instruction_specs.jsonl") if vector_specs else None,
        "failures": failures[:100],
        "failure_count": len(failures),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
