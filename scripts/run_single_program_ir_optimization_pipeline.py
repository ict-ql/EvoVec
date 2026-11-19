#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tasks import detect_language


def run_checked(command: list[str], *, dry_run: bool = False) -> None:
    print(display_command(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def display_command(command: list[str]) -> str:
    masked: list[str] = []
    hide_next = False
    for part in command:
        if hide_next:
            masked.append("***")
            hide_next = False
            continue
        masked.append(str(part))
        if part == "--api-key":
            hide_next = True
    return " ".join(shlex.quote(part) for part in masked)


def emit_or_copy_ir(args: argparse.Namespace, program: Path, kernel: str, ir_dir: Path) -> Path:
    ir_dir.mkdir(parents=True, exist_ok=True)
    out_ir = ir_dir / f"{kernel}.ll"
    if args.skip_existing_ir and out_ir.exists():
        return out_ir
    if program.suffix.lower() == ".ll":
        if program.resolve() != out_ir.resolve():
            shutil.copyfile(program, out_ir)
        return out_ir
    command = [
        args.clang,
        "-S",
        "-emit-llvm",
        f"-O{args.scalar_opt_level}",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-fno-unroll-loops",
        f"-march={args.target_attr_march}",
        *args.extra_clang_arg,
        str(program),
        "-o",
        str(out_ir),
    ]
    run_checked(command, dry_run=args.dry_run)
    return out_ir


def write_manifest(path: Path, *, kernel: str, program: Path, ir_file: Path, language: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "suite": "single",
        "task_id": f"single:{kernel}",
        "kernel": kernel,
        "root": str(program.parent),
        "source_file": str(program),
        "reference_kernel": str(program),
        "ir_file": str(ir_file),
        "language": language,
        "granularity": "translation_unit",
    }
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


def run_recommendations(args: argparse.Namespace, ir_file: Path, rec_dir: Path) -> Path:
    rec_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = rec_dir / "recommendations.jsonl"
    out_md = rec_dir / "recommendations.md"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "recommend_region_intrinsics.py"),
        "--ir-file",
        str(ir_file),
        "--checkpoint",
        str(args.checkpoint),
        "--instruction-graph-dir",
        str(args.instruction_graph_dir),
        "--out-jsonl",
        str(out_jsonl),
        "--out-md",
        str(out_md),
        "--top-k",
        str(args.top_k),
        "--device",
        args.device,
        "--candidate-modes",
        args.candidate_modes,
        "--store-slice-depth",
        str(args.store_slice_depth),
        "--store-slice-max-nodes",
        str(args.store_slice_max_nodes),
        "--max-report-regions",
        str(args.max_report_regions),
    ]
    if args.max_regions_per_function > 0:
        command.extend(["--max-regions-per-function", str(args.max_regions_per_function)])
    if args.include_memory_labels:
        command.append("--include-memory-labels")
    if args.only_predicted_change:
        command.extend(["--only-predicted-change", "--change-threshold", str(args.change_threshold)])
    run_checked(command, dry_run=args.dry_run)
    return out_jsonl


def resolve_intrinsic_declarations(
    args: argparse.Namespace,
    recommendations_jsonl: Path,
    declarations_dir: Path,
) -> Path | None:
    if args.intrinsic_declarations_json:
        return args.intrinsic_declarations_json
    if not args.llvm_project_root:
        return None
    declarations_dir.mkdir(parents=True, exist_ok=True)
    out_json = declarations_dir / "intrinsic_declarations.json"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "resolve_llvm_intrinsic_declarations.py"),
        "--llvm-project-root",
        str(args.llvm_project_root),
        "--recommendations-jsonl",
        str(recommendations_jsonl),
        "--recommendation-top-k",
        str(args.recommendation_top_k),
        "--out-json",
        str(out_json),
        "--cxx",
        args.cxx,
    ]
    if args.llvm_config:
        command.extend(["--llvm-config", str(args.llvm_config)])
    if args.intrinsic_decl_build_dir:
        command.extend(["--build-dir", str(args.intrinsic_decl_build_dir)])
    if args.keep_intrinsic_decl_build_dir:
        command.append("--keep-build-dir")
    run_checked(command, dry_run=args.dry_run)
    return out_json


def run_ir_generation(args: argparse.Namespace, manifest: Path, ir_dir: Path, recommendations_jsonl: Path, opt_dir: Path) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_tsvc_ir_optimization_experiment.py"),
        "--manifest",
        str(manifest),
        "--recommendations-jsonl",
        str(recommendations_jsonl),
        "--out-dir",
        str(opt_dir),
        "--api-base-url",
        args.api_base_url,
        "--api-key",
        args.api_key,
        "--model",
        args.model,
        "--max-tokens",
        str(args.max_tokens),
        "--limit",
        "1",
        "--modes",
        args.modes,
        "--clang",
        args.clang,
        "--cc",
        args.cc,
        "--ir-dir",
        str(ir_dir),
        "--skip-existing-ir",
        "--jobs",
        str(args.jobs),
        "--target",
        args.target,
        "--target-cpu",
        args.target_cpu,
        "--target-attr-march",
        args.target_attr_march,
        "--target-notes",
        args.target_notes,
        "--max-recommendation-regions",
        str(args.max_recommendation_regions),
        "--recommendation-top-k",
        str(args.recommendation_top_k),
    ]
    if args._resolved_intrinsic_declarations_json:
        command.extend(["--intrinsic-declarations-json", str(args._resolved_intrinsic_declarations_json)])
    if args.api_mode:
        command.extend(["--api-mode", args.api_mode])
    if args.temperature is not None:
        command.extend(["--temperature", str(args.temperature)])
    if args.generate_only:
        command.append("--generate-only")
    if args.skip_existing:
        command.append("--skip-existing")
    if args.no_ir_postprocess:
        command.append("--no-ir-postprocess")
    if args.no_target_attributes:
        command.append("--no-target-attributes")
    for extra in args.extra_clang_arg:
        command.extend(["--extra-clang-arg", extra])
    run_checked(command, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run region recommendation and guided IR generation for one source or LLVM IR program."
    )
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--kernel", help="Stable name for output files. Defaults to input file stem.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--instruction-graph-dir", type=Path, required=True)
    parser.add_argument(
        "--intrinsic-declarations-json",
        type=Path,
        help="Optional pre-resolved declarations file. Usually omit this for single-program runs.",
    )
    parser.add_argument(
        "--llvm-project-root",
        type=Path,
        help="If provided, resolve declarations from this program's recommendations after inference.",
    )
    parser.add_argument("--llvm-config", type=Path, help="llvm-config matching --llvm-project-root.")
    parser.add_argument("--cxx", default="c++", help="C++ compiler used by declaration resolver.")
    parser.add_argument("--intrinsic-decl-build-dir", type=Path)
    parser.add_argument("--keep-intrinsic-decl-build-dir", action="store_true")
    parser.add_argument("--api-base-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-mode", default="")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--modes", default="baseline,guided")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--candidate-modes", default="store-backward")
    parser.add_argument("--store-slice-depth", type=int, default=5)
    parser.add_argument("--store-slice-max-nodes", type=int, default=128)
    parser.add_argument("--max-regions-per-function", type=int, default=0)
    parser.add_argument("--max-report-regions", type=int, default=50)
    parser.add_argument("--include-memory-labels", action="store_true")
    parser.add_argument("--only-predicted-change", action="store_true")
    parser.add_argument("--change-threshold", type=float, default=0.0)
    parser.add_argument("--max-recommendation-regions", type=int, default=8)
    parser.add_argument("--recommendation-top-k", type=int, default=20)
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--scalar-opt-level", default="3", choices=["0", "1", "2", "3", "s", "z"])
    parser.add_argument("--target", default="x86_64-pc-linux-gnu")
    parser.add_argument("--target-cpu", default="auto")
    parser.add_argument("--target-attr-march", default="native")
    parser.add_argument("--target-notes", default="System V AMD64 ABI, optimize for native x86 vector execution.")
    parser.add_argument("--extra-clang-arg", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--skip-existing-ir", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--generate-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-ir-postprocess", action="store_true")
    parser.add_argument("--no-target-attributes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    program = args.program.resolve()
    if not program.exists():
        raise SystemExit(f"missing input program: {program}")
    if not program.is_file():
        raise SystemExit(f"input program is not a file: {program}")
    if args.jobs <= 0:
        raise SystemExit("--jobs must be positive")
    language = detect_language(program)
    if language is None:
        raise SystemExit(f"could not infer language from extension {program.suffix!r}")

    kernel = args.kernel or program.stem
    run_root = args.out_dir
    ir_dir = run_root / "prevec_ir"
    rec_dir = run_root / "recommendations" / "top20"
    decl_dir = run_root / "intrinsic_declarations"
    opt_dir = run_root / "ir_opt" / "gpt5_top20"
    manifest = run_root / "manifests" / "single_program.jsonl"

    ir_file = emit_or_copy_ir(args, program, kernel, ir_dir)
    write_manifest(manifest, kernel=kernel, program=program, ir_file=ir_file, language=language)
    recommendations_jsonl = run_recommendations(args, ir_file, rec_dir)
    args._resolved_intrinsic_declarations_json = resolve_intrinsic_declarations(args, recommendations_jsonl, decl_dir)
    run_ir_generation(args, manifest, ir_dir, recommendations_jsonl, opt_dir)

    print(json.dumps({
        "intrinsic_declarations_json": (
            str(args._resolved_intrinsic_declarations_json)
            if args._resolved_intrinsic_declarations_json
            else None
        ),
        "program": str(program),
        "manifest": str(manifest),
        "ir_file": str(ir_file),
        "recommendations_jsonl": str(recommendations_jsonl),
        "optimization_out_dir": str(opt_dir),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
