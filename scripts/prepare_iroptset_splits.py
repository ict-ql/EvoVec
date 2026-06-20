#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable


DEFAULT_DATASET = "YangziResearch/IR-OptSet"
DEFAULT_EXCLUDE_RE = r"(?i)(^|[/_.-])tsvc($|[/_.-])|llvm-test-suite|test-suite"
METADATA_COLUMNS = [
    "repo_name",
    "repo_file_path",
    "function_name",
    "repo_license",
    "structural_hash",
]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def require_module(module: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise SystemExit(f"missing Python module {module!r}; {install_hint}") from exc


def row_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(key, "")) for key in ("repo_name", "repo_file_path", "function_name"))


def keep_row(row: dict[str, Any], exclude_re: re.Pattern[str] | None) -> bool:
    if exclude_re and exclude_re.search(row_text(row)):
        return False
    return bool(row.get("repo_name"))


def iter_huggingface_rows(
    dataset: str,
    config: str | None,
    split: str,
    columns: list[str],
    token: str | bool | None,
) -> Iterable[tuple[int, dict[str, Any]]]:
    datasets = require_module(
        "datasets",
        "install it with: python3 -m pip install datasets",
    )
    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if config:
        kwargs["name"] = config
    if token is not None:
        kwargs["token"] = token
    stream = datasets.load_dataset(dataset, **kwargs)
    if hasattr(stream, "select_columns"):
        stream = stream.select_columns(columns)
    for index, row in enumerate(stream):
        yield index, dict(row)


def parquet_paths(parquet_dir: Path) -> list[Path]:
    paths = sorted(parquet_dir.glob("*.parquet"))
    if not paths:
        paths = sorted(parquet_dir.glob("**/*.parquet"))
    if not paths:
        raise SystemExit(f"no parquet files found under {parquet_dir}")
    return paths


def iter_parquet_rows(
    paths: list[Path],
    columns: list[str],
    batch_size: int,
) -> Iterable[tuple[int, dict[str, Any]]]:
    pq = require_module(
        "pyarrow.parquet",
        "install it with: python3 -m pip install pyarrow",
    )
    global_index = 0
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                yield global_index, row
                global_index += 1


def iter_source_rows(args: argparse.Namespace, columns: list[str]) -> Iterable[tuple[int, dict[str, Any]]]:
    if args.source == "huggingface":
        yield from iter_huggingface_rows(
            args.dataset,
            args.config,
            args.split,
            columns,
            True if args.hf_token else None,
        )
        return
    yield from iter_parquet_rows(parquet_paths(args.parquet_dir), columns, args.batch_size)


def collect_repo_indices(args: argparse.Namespace, exclude_re: re.Pattern[str] | None) -> dict[str, list[int]]:
    repo_to_indices: dict[str, list[int]] = defaultdict(list)
    seen = 0
    kept = 0
    for index, row in iter_source_rows(args, METADATA_COLUMNS):
        seen += 1
        if not keep_row(row, exclude_re):
            continue
        repo_to_indices[str(row["repo_name"])].append(index)
        kept += 1
        if args.scan_limit and kept >= args.scan_limit:
            break
    if not repo_to_indices:
        raise SystemExit("no rows remained after filtering")
    print(
        json.dumps(
            {
                "phase": "scan",
                "rows_seen": seen,
                "rows_kept": kept,
                "repos_kept": len(repo_to_indices),
            },
            sort_keys=True,
        )
    )
    return repo_to_indices


def choose_split_indices(
    repo_to_indices: dict[str, list[int]],
    targets: dict[str, int],
    seed: int,
) -> tuple[dict[str, list[int]], dict[str, list[str]]]:
    rng = random.Random(seed)
    repos = sorted(repo_to_indices)
    rng.shuffle(repos)

    selected: dict[str, list[int]] = {split: [] for split in targets}
    selected_repos: dict[str, list[str]] = {split: [] for split in targets}
    cursor = 0
    for split, target in targets.items():
        while len(selected[split]) < target and cursor < len(repos):
            repo = repos[cursor]
            cursor += 1
            indices = list(repo_to_indices[repo])
            rng.shuffle(indices)
            remaining = target - len(selected[split])
            take = indices[:remaining]
            if not take:
                continue
            selected[split].extend(take)
            selected_repos[split].append(repo)

    for split, target in targets.items():
        if len(selected[split]) < target:
            raise SystemExit(
                f"not enough filtered rows for {split}: requested {target}, selected {len(selected[split])}"
            )
        selected[split].sort()
    return selected, selected_repos


def write_selected_rows(
    args: argparse.Namespace,
    selected: dict[str, list[int]],
    fields: list[str],
) -> dict[str, int]:
    index_to_split = {
        index: split
        for split, indices in selected.items()
        for index in indices
    }
    wanted = set(index_to_split)
    counts = {split: 0 for split in selected}
    files = {
        split: (args.out_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in selected
    }
    columns = sorted(set(METADATA_COLUMNS + fields))
    try:
        for index, row in iter_source_rows(args, columns):
            split = index_to_split.get(index)
            if split is None:
                continue
            record = {
                "dataset": args.dataset if args.source == "huggingface" else str(args.parquet_dir),
                "dataset_index": index,
                "split": split,
            }
            for key in METADATA_COLUMNS + fields:
                record[key] = row.get(key)
            files[split].write(json.dumps(record, sort_keys=True) + "\n")
            counts[split] += 1
            if sum(counts.values()) == len(wanted):
                break
    finally:
        for file in files.values():
            file.close()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create repo-disjoint IR-OptSet train/val/test JSONL splits."
    )
    parser.add_argument("--source", choices=["huggingface", "parquet"], default="huggingface")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--hf-token",
        action="store_true",
        help="Pass token=True to datasets.load_dataset, using the locally configured Hugging Face token.",
    )
    parser.add_argument("--parquet-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/iroptset/splits/pilot_10k_v1"))
    parser.add_argument("--train-size", type=int, default=10_000)
    parser.add_argument("--val-size", type=int, default=2_000)
    parser.add_argument("--test-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--ir-fields",
        default="preprocessed_ir,o3_ir",
        help="Comma-separated IR fields to copy into split JSONL files.",
    )
    parser.add_argument(
        "--exclude-regex",
        default=DEFAULT_EXCLUDE_RE,
        help="Regex applied to repo/path/function text before sampling. Use '' to disable.",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=0,
        help="Optional number of filtered rows to scan before sampling; 0 scans all available rows.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    if args.source == "parquet" and not args.parquet_dir:
        parser.error("--parquet-dir is required when --source parquet")
    if min(args.train_size, args.val_size, args.test_size) < 0:
        parser.error("split sizes must be non-negative")

    fields = parse_csv(args.ir_fields)
    if not fields:
        parser.error("--ir-fields must contain at least one field")
    exclude_re = re.compile(args.exclude_regex) if args.exclude_regex else None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "train": args.train_size,
        "val": args.val_size,
        "test": args.test_size,
    }
    repo_to_indices = collect_repo_indices(args, exclude_re)
    selected, selected_repos = choose_split_indices(repo_to_indices, targets, args.seed)
    counts = write_selected_rows(args, selected, fields)

    plan = {
        "dataset": args.dataset if args.source == "huggingface" else str(args.parquet_dir),
        "source": args.source,
        "source_split": args.split,
        "seed": args.seed,
        "exclude_regex": args.exclude_regex,
        "ir_fields": fields,
        "targets": targets,
        "counts": counts,
        "repo_counts": {split: len(repos) for split, repos in selected_repos.items()},
        "selected_repos": selected_repos,
        "outputs": {split: str(args.out_dir / f"{split}.jsonl") for split in targets},
    }
    (args.out_dir / "split_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "write", "out_dir": str(args.out_dir), "counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
