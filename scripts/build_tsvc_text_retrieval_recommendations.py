#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile


TOKEN_RE = re.compile(r"[A-Za-z]+|\d+")


def clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


def load_aliases(path: Path) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    aliases: dict[str, list[str]] = {}
    for c_intrinsic, values in raw.items():
        if isinstance(values, str):
            values = [values]
        aliases[str(c_intrinsic)] = [str(value) for value in values if str(value).startswith("llvm.")]
    return aliases


def build_corpus(xml_path: Path, alias_path: Path) -> list[dict[str, object]]:
    aliases = load_aliases(alias_path)
    grouped: dict[str, dict[str, object]] = {}
    for element in ET.parse(xml_path).getroot().findall("intrinsic"):
        c_intrinsic = str(element.attrib.get("name", "")).strip()
        llvm_intrinsics = aliases.get(c_intrinsic, [])
        if not llvm_intrinsics:
            continue
        description = clean_text(element.findtext("description"))
        for llvm_intrinsic in llvm_intrinsics:
            row = grouped.setdefault(
                llvm_intrinsic,
                {
                    "intrinsic": llvm_intrinsic,
                    "c_intrinsics": [],
                    "descriptions": [],
                },
            )
            c_intrinsics = row["c_intrinsics"]
            descriptions = row["descriptions"]
            assert isinstance(c_intrinsics, list)
            assert isinstance(descriptions, list)
            if c_intrinsic not in c_intrinsics:
                c_intrinsics.append(c_intrinsic)
            if description and description not in descriptions:
                descriptions.append(description)

    corpus: list[dict[str, object]] = []
    for intrinsic, row in sorted(grouped.items()):
        c_intrinsics = list(row["c_intrinsics"])
        descriptions = list(row["descriptions"])
        document = " ".join([intrinsic, *c_intrinsics, *descriptions])
        corpus.append(
            {
                "intrinsic": intrinsic,
                "c_intrinsics": c_intrinsics,
                "descriptions": descriptions,
                "document": document,
            }
        )
    return corpus


def summary_kernel(filename: str) -> str:
    name = PurePosixPath(filename.replace("\\", "/")).name
    if name.endswith(".summary.txt"):
        return name[: -len(".summary.txt")]
    return Path(name).stem


def load_summaries(path: Path) -> dict[str, tuple[str, str]]:
    summaries: dict[str, tuple[str, str]] = {}
    if path.is_file() and path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith("/") or not name.lower().endswith(".txt"):
                    continue
                kernel = summary_kernel(name)
                if kernel in summaries:
                    raise SystemExit(f"duplicate summary for kernel {kernel}: {name}")
                summaries[kernel] = (name, archive.read(name).decode("utf-8", errors="replace"))
        return summaries
    if path.is_dir():
        files = sorted(path.rglob("*.summary.txt"))
        if not files:
            files = sorted(path.rglob("*.txt"))
        for file_path in files:
            kernel = summary_kernel(file_path.name)
            if kernel in summaries:
                raise SystemExit(f"duplicate summary for kernel {kernel}: {file_path}")
            summaries[kernel] = (str(file_path), file_path.read_text(encoding="utf-8", errors="replace"))
        return summaries
    raise SystemExit(f"summary input must be a directory or zip file: {path}")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(document)) for document in documents]
        self.lengths = [sum(frequencies.values()) for frequencies in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        document_frequencies: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            document_frequencies.update(frequencies.keys())
        count = len(documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def scores(self, query: str) -> list[float]:
        query_terms = set(tokenize(query))
        scores: list[float] = []
        for frequencies, length in zip(self.term_frequencies, self.lengths):
            score = 0.0
            length_ratio = length / self.average_length if self.average_length else 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (1.0 - self.b + self.b * length_ratio)
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1.0) / denominator
            scores.append(score)
        return scores


def stable_random(seed: int, kernel: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{kernel}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def rank_corpus(
    method: str,
    kernel: str,
    query: str,
    corpus: list[dict[str, object]],
    bm25: BM25,
    top_k: int,
    seed: int,
) -> list[tuple[int, float]]:
    if method == "bm25":
        scores = bm25.scores(query)
        ranked = sorted(
            range(len(corpus)),
            key=lambda index: (-scores[index], str(corpus[index]["intrinsic"])),
        )
        return [(index, scores[index]) for index in ranked[:top_k]]
    if method == "random":
        indices = list(range(len(corpus)))
        stable_random(seed, kernel).shuffle(indices)
        return [(index, 0.0) for index in indices[:top_k]]
    return [(index, 0.0) for index in range(len(corpus))]


def recommendation_item(row: dict[str, object], rank: int, score: float) -> dict[str, object]:
    descriptions = list(row["descriptions"])
    return {
        "rank": rank,
        "label": row["intrinsic"],
        "intrinsic": row["intrinsic"],
        "score": score,
        "description": descriptions[0] if descriptions else "",
        "source": "intel_intrinsics_guide_xml",
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build TSVC BM25, random, or full-list recommendations from Intel XML descriptions."
    )
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--alias-map", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--corpus-out", type=Path)
    parser.add_argument("--names-out", type=Path)
    parser.add_argument("--method", choices=["bm25", "random", "full"], default="bm25")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    corpus = build_corpus(args.xml, args.alias_map)
    if not corpus:
        raise SystemExit("the alias map and XML produced an empty retrieval corpus")
    summaries = load_summaries(args.summaries)
    if not summaries:
        raise SystemExit(f"no summaries found in {args.summaries}")
    effective_top_k = len(corpus) if args.method == "full" else min(args.top_k, len(corpus))
    bm25 = BM25([str(row["document"]) for row in corpus], k1=args.bm25_k1, b=args.bm25_b)

    output_rows: list[dict[str, object]] = []
    for kernel, (source_file, summary) in sorted(summaries.items()):
        ranked = rank_corpus(args.method, kernel, summary, corpus, bm25, effective_top_k, args.seed)
        output_rows.append(
            {
                "source_file": source_file,
                "function_name": kernel,
                "region_index": 0,
                "change_probability": 1.0,
                "query_summary": summary,
                "retrieval_method": args.method,
                "recommendations": [
                    recommendation_item(corpus[index], rank, score)
                    for rank, (index, score) in enumerate(ranked, start=1)
                ],
            }
        )

    write_jsonl(args.out_jsonl, output_rows)
    corpus_out = args.corpus_out or args.out_jsonl.with_suffix(".corpus.json")
    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    corpus_out.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    names_out = args.names_out or args.out_jsonl.with_suffix(".names.txt")
    names_out.parent.mkdir(parents=True, exist_ok=True)
    names_out.write_text(
        "".join(f"{row['intrinsic']}\n" for row in corpus),
        encoding="utf-8",
    )
    summary = {
        "method": args.method,
        "corpus_entries": len(corpus),
        "queries": len(output_rows),
        "recommendations_per_query": effective_top_k,
        "seed": args.seed if args.method == "random" else None,
        "out_jsonl": str(args.out_jsonl),
        "corpus_out": str(corpus_out),
        "names_out": str(names_out),
    }
    summary_path = args.out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
