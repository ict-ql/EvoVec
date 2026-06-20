#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any


INTRINSIC_CALL_RE = re.compile(
    r'\b(?:call|invoke|callbr)\b[^@\n;]*@(?:"(?P<quoted>llvm\.[^"]+)"|(?P<bare>llvm\.[A-Za-z0-9_.$-]+))'
)

DEFAULT_EXCLUDE_INTRINSIC_PREFIXES = (
    "llvm.dbg.",
    "llvm.lifetime.",
    "llvm.invariant.",
    "llvm.assume",
    "llvm.expect",
    "llvm.donothing",
    "llvm.sideeffect",
    "llvm.annotation.",
    "llvm.ptr.annotation.",
    "llvm.var.annotation",
    "llvm.objectsize.",
    "llvm.is.constant.",
    "llvm.experimental.noalias.scope.",
    "llvm.experimental.deoptimize.",
    "llvm.experimental.guard.",
    "llvm.type.test",
    "llvm.type.checked.load",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def record_ir(row: dict[str, Any], field: str) -> str | None:
    value = row.get(field)
    if value is None and isinstance(row.get("ir"), dict):
        value = row["ir"].get(field)
    return str(value) if value else None


def split_inputs(args: argparse.Namespace) -> list[Path]:
    if args.split_jsonl:
        return args.split_jsonl
    if not args.split_dir:
        raise SystemExit("provide --split-dir or one or more --split-jsonl paths")
    paths = [args.split_dir / f"{split}.jsonl" for split in ("train", "val", "test")]
    return [path for path in paths if path.exists()]


def split_name(path: Path) -> str:
    name = path.name
    if name.endswith(".jsonl"):
        return name[:-6]
    return path.stem


def parse_prefixes(values: list[str] | None) -> tuple[str, ...]:
    prefixes: list[str] = []
    for value in values or []:
        for part in value.split(","):
            prefix = part.strip()
            if prefix:
                prefixes.append(prefix.lower())
    return tuple(dict.fromkeys(prefixes))


def strip_ir_comment(line: str) -> str:
    if ";" in line:
        line = line.split(";", 1)[0]
    return line


def intrinsic_is_allowed(
    intrinsic: str,
    include_prefixes: tuple[str, ...],
    exclude_prefixes: tuple[str, ...],
) -> bool:
    lowered = intrinsic.lower()
    if include_prefixes and not any(lowered.startswith(prefix) for prefix in include_prefixes):
        return False
    return not any(lowered.startswith(prefix) for prefix in exclude_prefixes)


def extract_intrinsic_counts(
    ir_text: str,
    include_prefixes: tuple[str, ...] = (),
    exclude_prefixes: tuple[str, ...] = DEFAULT_EXCLUDE_INTRINSIC_PREFIXES,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for raw_line in ir_text.splitlines():
        line = strip_ir_comment(raw_line)
        if not line:
            continue
        for match in INTRINSIC_CALL_RE.finditer(line):
            intrinsic = (match.group("quoted") or match.group("bare") or "").lower()
            if intrinsic and intrinsic_is_allowed(intrinsic, include_prefixes, exclude_prefixes):
                counts[intrinsic] += 1
    return counts


def resolve_ir_field(args: argparse.Namespace) -> str:
    if args.ir_field and args.compile_ir_field and args.ir_field != args.compile_ir_field:
        raise SystemExit("--ir-field and --compile-ir-field disagree; pass only one IR field")
    return args.ir_field or args.compile_ir_field or "o3_ir"


def warn_deprecated_backend_args(args: argparse.Namespace) -> None:
    if args.backend != "intrinsic":
        print(
            f"warning: --backend={args.backend!r} is deprecated and ignored; "
            "labels are extracted directly from LLVM IR intrinsics",
            file=sys.stderr,
        )
    deprecated_values = {
        "--llc": args.llc,
        "--target": args.target,
        "--mcpu": args.mcpu,
        "--mattr": args.mattr,
        "--opt-level": args.opt_level,
        "--timeout": args.timeout,
    }
    used = [
        name
        for name, value in deprecated_values.items()
        if value not in {None, "", "llc", "x86_64-pc-linux-gnu", "native", "-O3", 60}
    ]
    if args.keep_artifacts:
        used.append("--keep-artifacts")
    if used:
        print(
            "warning: backend compilation options are deprecated and ignored: " + ", ".join(used),
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract LLVM intrinsic instruction labels from IR-OptSet split records."
    )
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--split-jsonl", type=Path, nargs="*")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/iroptset/instruction_labels/pilot_10k_v1"))
    parser.add_argument("--ir-field", help="IR field to scan for LLVM intrinsic calls; defaults to o3_ir.")
    parser.add_argument("--compile-ir-field", help="Deprecated alias for --ir-field.")
    parser.add_argument(
        "--include-prefix",
        action="append",
        help="Only keep intrinsic names with this prefix. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        help="Drop intrinsic names with this prefix in addition to the default metadata/noise filters.",
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Keep debug/lifetime/control intrinsics instead of applying the default noise filters.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional per-split row limit; 0 means all rows.")

    # Backward-compatible no-op options kept so old pipeline commands do not
    # accidentally force an LLVM backend just to build labels.
    parser.add_argument(
        "--backend",
        choices=["intrinsic", "auto", "llc", "llvmlite"],
        default="intrinsic",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--llc", default="llc", help=argparse.SUPPRESS)
    parser.add_argument("--target", default="x86_64-pc-linux-gnu", help=argparse.SUPPRESS)
    parser.add_argument("--mcpu", default="native", help=argparse.SUPPRESS)
    parser.add_argument("--mattr", default="", help=argparse.SUPPRESS)
    parser.add_argument("--opt-level", default="-O3", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=int, default=60, help=argparse.SUPPRESS)
    parser.add_argument("--keep-artifacts", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    warn_deprecated_backend_args(args)
    ir_field = resolve_ir_field(args)
    include_prefixes = parse_prefixes(args.include_prefix)
    extra_exclude_prefixes = parse_prefixes(args.exclude_prefix)
    default_exclude_prefixes = () if args.no_default_excludes else DEFAULT_EXCLUDE_INTRINSIC_PREFIXES
    exclude_prefixes = tuple(dict.fromkeys((*default_exclude_prefixes, *extra_exclude_prefixes)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for input_path in split_inputs(args):
        split = split_name(input_path)
        rows = load_jsonl(input_path)
        if args.limit > 0:
            rows = rows[: args.limit]
        labels_path = args.out_dir / f"{split}.labels.jsonl"
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        ok = 0
        failed = 0
        empty = 0
        instruction_vocab: Counter[str] = Counter()
        with labels_path.open("w", encoding="utf-8") as labels_file:
            for row in rows:
                ir_text = record_ir(row, ir_field)
                if ir_text is None:
                    failed += 1
                    labels_file.write(
                        json.dumps(
                            {
                                "ok": False,
                                "error": f"missing IR field {ir_field}",
                                "dataset_index": row.get("dataset_index"),
                                "split": split,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    continue

                counts = extract_intrinsic_counts(ir_text, include_prefixes, exclude_prefixes)
                if not counts:
                    empty += 1
                instruction_vocab.update(counts)
                ok += 1
                count_map = dict(sorted(counts.items()))
                label_record = {
                    "ok": True,
                    "split": split,
                    "dataset_index": row.get("dataset_index"),
                    "repo_name": row.get("repo_name"),
                    "repo_file_path": row.get("repo_file_path"),
                    "function_name": row.get("function_name"),
                    "structural_hash": row.get("structural_hash"),
                    "label_kind": "llvm_intrinsic",
                    "label_source": "llvm_ir",
                    "ir_field": ir_field,
                    "include_prefixes": list(include_prefixes),
                    "exclude_prefixes": list(exclude_prefixes),
                    "instruction_counts": count_map,
                    "instructions": sorted(counts),
                    "instruction_total": sum(counts.values()),
                    "instruction_unique": len(counts),
                    "intrinsic_counts": count_map,
                    "intrinsics": sorted(counts),
                    "intrinsic_total": sum(counts.values()),
                    "intrinsic_unique": len(counts),
                }
                labels_file.write(json.dumps(label_record, sort_keys=True) + "\n")
        summaries[split] = {
            "input": str(input_path),
            "labels": str(labels_path),
            "rows": len(rows),
            "ok": ok,
            "failed": failed,
            "empty": empty,
            "instruction_vocab_size": len(instruction_vocab),
            "top_instructions": instruction_vocab.most_common(50),
        }
        print(json.dumps({"split": split, **summaries[split]}, sort_keys=True))

    summary = {
        "label_kind": "llvm_intrinsic",
        "label_source": "llvm_ir",
        "ir_field": ir_field,
        "include_prefixes": list(include_prefixes),
        "exclude_prefixes": list(exclude_prefixes),
        "splits": summaries,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if all(item["failed"] == 0 for item in summaries.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
