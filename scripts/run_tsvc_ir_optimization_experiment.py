#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
DEFAULT_TEMPLATE = REPO_ROOT / "prompts/llvm_ir_vectorize_fast_v1.txt"
DEFAULT_REPAIR_TEMPLATE = REPO_ROOT / "prompts/llvm_ir_self_correction_fast_v1.txt"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"expected object row in {path}:{line_number}")
            rows.append(row)
    return rows


def run_cmd(cmd: list[str], *, timeout: int | None = None, input_text: str | None = None) -> dict[str, Any]:
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_sec": round(time.time() - start, 3),
            "ok": proc.returncode == 0,
        }
    except FileNotFoundError as exc:
        return {
            "cmd": cmd,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
            "elapsed_sec": round(time.time() - start, 3),
            "ok": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "cmd": cmd,
            "returncode": 124,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed_sec": round(time.time() - start, 3),
            "ok": False,
            "error": "timeout",
        }


def parse_tsvc_output(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return {"ok": False, "error": "empty stdout"}
    data_line = lines[-1]
    columns = data_line.split()
    if len(columns) < 3:
        return {"ok": False, "error": f"could not parse TSVC line: {data_line!r}"}
    try:
        time_sec = float(columns[-2])
        checksum = float(columns[-1])
        if not math.isfinite(time_sec) or time_sec < 0.0:
            return {"ok": False, "error": f"invalid TSVC time: {columns[-2]!r}", "line": data_line}
        return {
            "ok": True,
            "line": data_line,
            "time_sec": time_sec,
            "checksum": checksum,
            "stdout": stdout,
        }
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "line": data_line}


def strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip() + "\n"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def successful_existing_generation(llm_path: Path, candidate_ir_path: Path) -> dict[str, Any] | None:
    if not llm_path.exists() or not candidate_ir_path.exists() or candidate_ir_path.stat().st_size == 0:
        return None
    try:
        llm_result = json.loads(llm_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(llm_result, dict) or not llm_result.get("ok"):
        return None
    return llm_result


def select_tasks(tasks: list[dict[str, Any]], kernels: str, offset: int, limit: int) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(tasks))
    if kernels.strip().lower() != "all":
        wanted = [item.strip() for item in kernels.split(",") if item.strip()]
        wanted_set = set(wanted)
        indexed = [(index, task) for index, task in indexed if str(task.get("kernel")) in wanted_set]
        missing = sorted(wanted_set.difference(str(task.get("kernel")) for _, task in indexed))
        if missing:
            raise SystemExit(f"kernels not found in manifest: {', '.join(missing)}")
    indexed = indexed[offset:]
    return indexed if limit <= 0 else indexed[:limit]


def resolve_tsvc_root(args: argparse.Namespace, task: dict[str, Any]) -> Path:
    if args.tsvc_root:
        return args.tsvc_root
    root = Path(str(task.get("root", "")))
    if root.exists():
        return root
    return REPO_ROOT / "runs/tsvc_kernels/v1"


def resolve_task_path(task: dict[str, Any], key: str, fallback: Path) -> Path:
    raw = str(task.get(key, "")).strip()
    if raw:
        path = Path(raw)
        if path.exists():
            return path
    return fallback


def resolve_existing_path(raw: Any) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.exists() else None


def task_paths(args: argparse.Namespace, task: dict[str, Any]) -> dict[str, Path]:
    kernel = str(task["kernel"])
    root = resolve_tsvc_root(args, task)
    kernel_dir = root / "kernels" / kernel
    return {
        "root": root,
        "support_dir": resolve_task_path(task, "support_dir", root / "support"),
        "reference_kernel": resolve_task_path(task, "reference_kernel", kernel_dir / "reference_kernel.c"),
        "harness": resolve_task_path(task, "harness", kernel_dir / "harness.c"),
    }


def compile_scalar_ir(args: argparse.Namespace, task: dict[str, Any], ir_path: Path) -> dict[str, Any]:
    ir_path.parent.mkdir(parents=True, exist_ok=True)
    input_ir = resolve_existing_path(task.get("ir_file"))
    if input_ir is not None:
        if input_ir.resolve() != ir_path.resolve():
            shutil.copyfile(input_ir, ir_path)
            return {"ok": True, "copied_from": str(input_ir), "ir_file": str(ir_path)}
        return {"ok": True, "reused": True, "ir_file": str(ir_path)}
    if args.skip_existing_ir and ir_path.exists():
        return {"ok": True, "skipped": True, "ir_file": str(ir_path)}
    if str(task.get("suite", "")).strip().lower() == "single":
        source = resolve_existing_path(task.get("source_file")) or resolve_existing_path(task.get("reference_kernel"))
        if source is None:
            return {
                "ok": False,
                "error": "single task requires an existing ir_file or source_file/reference_kernel",
                "task": task,
            }
        cmd = [
            args.clang,
            "-S",
            "-emit-llvm",
            f"-O{args.scalar_opt_level}",
            "-fno-vectorize",
            "-fno-slp-vectorize",
            "-fno-unroll-loops",
            f"-march={args.target_attr_march}",
            *args.extra_clang_arg,
            str(source),
            "-o",
            str(ir_path),
        ]
        return run_cmd(cmd, timeout=args.build_timeout)
    paths = task_paths(args, task)
    cmd = [
        args.clang,
        "-S",
        "-emit-llvm",
        f"-O{args.scalar_opt_level}",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-fno-unroll-loops",
        f"-march={args.target_attr_march}",
        "-I",
        str(paths["support_dir"]),
        str(paths["reference_kernel"]),
        "-o",
        str(ir_path),
    ]
    return run_cmd(cmd, timeout=args.build_timeout)


def build_reference(args: argparse.Namespace, task: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    paths = task_paths(args, task)
    out_dir.mkdir(parents=True, exist_ok=True)
    support = paths["support_dir"]
    exe = out_dir / "reference.exe"
    cmd = [
        args.cc,
        "-O3",
        f"-march={args.target_attr_march}",
        "-no-pie",
        "-I",
        str(support),
        str(paths["reference_kernel"]),
        str(support / "common.c"),
        str(support / "dummy.c"),
        str(support / "globals.c"),
        str(paths["harness"]),
        "-lm",
        "-o",
        str(exe),
    ]
    compile_result = run_cmd(cmd, timeout=args.build_timeout)
    run_result: dict[str, Any] | None = None
    parsed: dict[str, Any] | None = None
    if compile_result["ok"]:
        run_result = run_cmd([str(exe)], timeout=args.run_timeout)
        parsed = parse_tsvc_output(str(run_result["stdout"])) if run_result["ok"] else None
    return {
        "exe": str(exe),
        "compile": compile_result,
        "run": run_result,
        "parsed": parsed,
        "ok": bool(compile_result["ok"] and run_result and run_result["ok"] and parsed and parsed["ok"]),
    }


def build_and_run_ir(
    args: argparse.Namespace,
    task: dict[str, Any],
    ir_path: Path,
    out_dir: Path,
    expected_checksum: float,
) -> dict[str, Any]:
    paths = task_paths(args, task)
    out_dir.mkdir(parents=True, exist_ok=True)
    support = paths["support_dir"]
    exe = out_dir / "candidate.exe"
    cmd = [
        args.clang,
        "-O3",
        f"-march={args.target_attr_march}",
        "-no-pie",
        "-I",
        str(support),
        *args.extra_clang_arg,
        str(ir_path),
        str(support / "common.c"),
        str(support / "dummy.c"),
        str(support / "globals.c"),
        str(paths["harness"]),
        "-lm",
        "-o",
        str(exe),
    ]
    compile_result = run_cmd(cmd, timeout=args.build_timeout)
    run_result: dict[str, Any] | None = None
    parsed: dict[str, Any] | None = None
    checksum_ok = False
    if compile_result["ok"]:
        run_result = run_cmd([str(exe)], timeout=args.run_timeout)
        parsed = parse_tsvc_output(str(run_result["stdout"])) if run_result["ok"] else None
        if parsed and parsed["ok"]:
            checksum_ok = math.isclose(
                float(parsed["checksum"]),
                expected_checksum,
                rel_tol=0.0,
                abs_tol=args.checksum_atol,
            )
    return {
        "method": "reference-oracle-checksum-differential",
        "exe": str(exe),
        "compile": compile_result,
        "run": run_result,
        "parsed": parsed,
        "expected_checksum": expected_checksum,
        "checksum_atol": args.checksum_atol,
        "checksum_ok": checksum_ok,
        "ok": bool(compile_result["ok"] and run_result and run_result["ok"] and parsed and parsed["ok"] and checksum_ok),
    }


MODULE_DIRECTIVE_PATTERN = re.compile(r'^(target (?:datalayout|triple)\s*=\s*"[^"]*")$', re.MULTILINE)
METADATA_ATTACHMENT_PATTERN = re.compile(r",\s*![A-Za-z_][A-Za-z0-9_.-]*\s+![0-9]+")


def synchronize_module_target(source_ir: str, candidate_ir_path: Path) -> dict[str, Any]:
    source_directives = {
        directive.split("=", 1)[0].strip(): directive
        for directive in MODULE_DIRECTIVE_PATTERN.findall(source_ir)
    }
    candidate_ir = candidate_ir_path.read_text(encoding="utf-8", errors="replace")
    changes: list[dict[str, str | None]] = []
    for key, source_directive in source_directives.items():
        pattern = re.compile(rf'^{re.escape(key)}\s*=\s*"[^"]*"$', re.MULTILINE)
        match = pattern.search(candidate_ir)
        if match:
            old_directive = match.group(0)
            if old_directive != source_directive:
                candidate_ir = pattern.sub(source_directive, candidate_ir, count=1)
                changes.append({"directive": key, "old": old_directive, "new": source_directive})
        else:
            candidate_ir = f"{source_directive}\n{candidate_ir}"
            changes.append({"directive": key, "old": None, "new": source_directive})
    if changes:
        candidate_ir_path.write_text(candidate_ir, encoding="utf-8")
    return {"changed": bool(changes), "changes": changes, "path": str(candidate_ir_path)}


def normalize_ir_for_alive2(ir_text: str) -> tuple[str, dict[str, int]]:
    output_lines: list[str] = []
    removed_metadata_definitions = 0
    removed_attachments = 0
    for line in ir_text.splitlines():
        if line.lstrip().startswith("!"):
            removed_metadata_definitions += 1
            continue
        normalized_line, count = METADATA_ATTACHMENT_PATTERN.subn("", line)
        removed_attachments += count
        output_lines.append(normalized_line)
    return "\n".join(output_lines) + "\n", {
        "removed_metadata_definitions": removed_metadata_definitions,
        "removed_metadata_attachments": removed_attachments,
    }


def run_alive2(
    args: argparse.Namespace,
    source_ir_path: Path,
    candidate_ir_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    if not args.alive2:
        return {"enabled": False, "ok": True, "skipped": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized_source, source_normalization = normalize_ir_for_alive2(
        source_ir_path.read_text(encoding="utf-8", errors="replace")
    )
    normalized_candidate, candidate_normalization = normalize_ir_for_alive2(
        candidate_ir_path.read_text(encoding="utf-8", errors="replace")
    )
    normalized_source_path = out_dir / "source.alive2.ll"
    normalized_candidate_path = out_dir / "candidate.alive2.ll"
    normalized_source_path.write_text(normalized_source, encoding="utf-8")
    normalized_candidate_path.write_text(normalized_candidate, encoding="utf-8")
    command = [
        args.alive_tv,
        *args.alive2_arg,
        str(normalized_source_path),
        str(normalized_candidate_path),
    ]
    result = run_cmd(command, timeout=args.alive2_timeout)
    output = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
    success_markers = (
        "Transformation seems to be correct!",
        "0 incorrect transformations",
    )
    has_failure = bool(
        "ERROR:" in output
        or "doesn't verify" in output
        or re.search(r"\b[1-9]\d* incorrect transformations", output)
        or re.search(r"\b[1-9]\d* failed-to-prove transformations", output)
    )
    has_counterexample = bool(
        "doesn't verify" in output
        or re.search(r"\b[1-9]\d* incorrect transformations", output)
    )
    result["enabled"] = True
    result["source_ir"] = str(normalized_source_path)
    result["candidate_ir"] = str(normalized_candidate_path)
    result["normalization"] = {
        "source": source_normalization,
        "candidate": candidate_normalization,
    }
    result["verified"] = bool(
        result["ok"]
        and any(marker in output for marker in success_markers)
        and not has_failure
    )
    result["ok"] = result["verified"]
    result["status"] = "pass" if result["verified"] else ("fail" if has_counterexample else "inconclusive")
    return result


def validate_ir_candidate(
    args: argparse.Namespace,
    task: dict[str, Any],
    source_ir_path: Path,
    candidate_ir_path: Path,
    out_dir: Path,
    expected_checksum: float,
) -> dict[str, Any]:
    alive2_result = run_alive2(args, source_ir_path, candidate_ir_path, out_dir / "alive2")
    differential_result = build_and_run_ir(args, task, candidate_ir_path, out_dir, expected_checksum)
    if args.differential_fuzz:
        from neural_asm_pipeline.differential_fuzz import run_differential_fuzz

        fuzz_result = run_differential_fuzz(
            source_ir_path=source_ir_path,
            candidate_ir_path=candidate_ir_path,
            task=task,
            support_dir=task_paths(args, task)["support_dir"],
            out_dir=out_dir / "libfuzzer",
            clang=args.clang,
            llvm_link=args.llvm_link,
            march=args.target_attr_march,
            runs=args.fuzz_runs,
            max_len=args.fuzz_max_len,
            per_input_timeout=args.fuzz_input_timeout,
            timeout=args.fuzz_timeout,
            sanitizers=args.fuzz_sanitizers,
            extra_clang_args=args.extra_clang_arg,
            command_runner=run_cmd,
        )
    else:
        fuzz_result = {"enabled": False, "ok": True, "status": "disabled", "skipped": True}
    methods = ["reference-oracle-checksum-differential"]
    if args.alive2:
        methods.insert(0, "alive2")
    if args.differential_fuzz:
        methods.append("libfuzzer-merged-ir-differential")
    semantic_statuses: list[str] = []
    if args.alive2:
        semantic_statuses.append(str(alive2_result.get("status") or "inconclusive"))
    if args.differential_fuzz:
        semantic_statuses.append(str(fuzz_result.get("status") or "inconclusive"))
    has_semantic_failure = "fail" in semantic_statuses
    has_semantic_pass = "pass" in semantic_statuses
    tsvc_ok = bool(differential_result["ok"])
    accepted = (
        tsvc_ok and has_semantic_pass and not has_semantic_failure
        if semantic_statuses
        else tsvc_ok
    )
    repairable = bool(
        has_semantic_failure
        or not differential_result.get("compile", {}).get("ok")
        or not differential_result.get("run", {}).get("ok")
        or not differential_result.get("checksum_ok")
    )
    return {
        "methods": methods,
        "acceptance_policy": "tsvc-pass-and-semantic-pass-without-counterexample" if semantic_statuses else "tsvc-checksum",
        "semantic_statuses": semantic_statuses,
        "repairable": repairable,
        "alive2": alive2_result,
        "differential": differential_result,
        "differential_fuzz": fuzz_result,
        "ok": accepted,
    }


def repair_verifier_feedback(result: dict[str, Any]) -> dict[str, Any]:
    alive2_result = result.get("alive2") or {}
    differential = result.get("differential") or result
    compile_result = differential.get("compile") or {}
    run_result = differential.get("run") or {}
    parsed = differential.get("parsed") or {}
    fuzz_result = result.get("differential_fuzz") or {}
    fuzz_stage = str(fuzz_result.get("stage") or "")
    fuzz_stage_result = (fuzz_result.get("stages") or {}).get(fuzz_stage) or {}
    return {
        "ok": bool(result.get("ok")),
        "acceptance_policy": result.get("acceptance_policy"),
        "repairable": result.get("repairable"),
        "alive2": {
            "enabled": bool(alive2_result.get("enabled")),
            "ok": bool(alive2_result.get("ok")),
            "status": alive2_result.get("status"),
            "returncode": alive2_result.get("returncode"),
            "stdout": alive2_result.get("stdout", ""),
            "stderr": alive2_result.get("stderr", ""),
            "error": alive2_result.get("error"),
        },
        "differential": {
            "ok": bool(differential.get("ok")),
            "compile_ok": bool(compile_result.get("ok")),
            "compile_returncode": compile_result.get("returncode"),
            "compile_stderr": compile_result.get("stderr", ""),
            "compile_error": compile_result.get("error"),
            "run_ok": bool(run_result.get("ok")),
            "run_returncode": run_result.get("returncode"),
            "run_stdout": run_result.get("stdout", ""),
            "run_stderr": run_result.get("stderr", ""),
            "run_error": run_result.get("error"),
            "parse_error": parsed.get("error"),
            "expected_checksum": differential.get("expected_checksum"),
            "observed_checksum": parsed.get("checksum"),
            "checksum_atol": differential.get("checksum_atol"),
            "checksum_ok": bool(differential.get("checksum_ok")),
        },
        "differential_fuzz": {
            "enabled": bool(fuzz_result.get("enabled", bool(fuzz_result))),
            "ok": bool(fuzz_result.get("ok")),
            "status": fuzz_result.get("status"),
            "stage": fuzz_stage or None,
            "returncode": fuzz_stage_result.get("returncode"),
            "stdout": fuzz_stage_result.get("stdout", ""),
            "stderr": fuzz_stage_result.get("stderr", ""),
            "error": fuzz_result.get("error") or fuzz_stage_result.get("error"),
        },
    }


def patch_target_attributes_if_enabled(args: argparse.Namespace, ir_path: Path) -> dict[str, Any]:
    if args.no_target_attributes:
        return {"patched": False, "disabled": True, "path": str(ir_path)}
    from neural_asm_pipeline.target_attributes import patch_ir_file

    return patch_ir_file(ir_path, args.target_attributes)


def postprocess_ir_if_enabled(args: argparse.Namespace, ir_path: Path) -> dict[str, Any]:
    if args.no_ir_postprocess:
        return {"changed": False, "disabled": True, "path": str(ir_path)}
    from neural_asm_pipeline.ir_postprocess import postprocess_ir_file

    return postprocess_ir_file(ir_path)


def recommendation_key(row: dict[str, Any]) -> str:
    function_name = str(row.get("function_name", "")).strip()
    if function_name:
        return function_name
    source_file = str(row.get("source_file", "")).strip()
    return Path(source_file).stem if source_file else ""


def load_recommendations(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if path is None:
        return grouped
    for row in load_jsonl(path):
        key = recommendation_key(row)
        if key:
            grouped[key].append(row)
    return grouped


def recommendations_for_task(
    grouped: dict[str, list[dict[str, Any]]],
    task: dict[str, Any],
    kernel: str,
) -> list[dict[str, Any]]:
    rows = grouped.get(kernel, [])
    if rows:
        return rows
    if str(task.get("suite", "")).strip().lower() != "single":
        return []
    output: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        output.extend(group_rows)
    return output


def clean_intrinsic_declaration(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("declare "):
            continue
        return re.sub(r"\s+#\d+\s*$", "", stripped)
    return " ".join(text.split())


def load_intrinsic_declarations(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if path.suffix.lower() == ".jsonl":
        rows = load_jsonl(path)
        return {
            str(row["name"]): clean_intrinsic_declaration(str(row["declaration"]))
            for row in rows
            if row.get("name") and str(row.get("declaration", "")).strip()
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("declarations"), dict):
        return {
            str(name): clean_intrinsic_declaration(str(declaration))
            for name, declaration in data["declarations"].items()
            if str(declaration).strip()
        }
    if isinstance(data, dict):
        return {
            str(name): clean_intrinsic_declaration(str(declaration))
            for name, declaration in data.items()
            if isinstance(declaration, str) and declaration.strip()
        }
    if isinstance(data, list):
        return {
            str(row["name"]): clean_intrinsic_declaration(str(row["declaration"]))
            for row in data
            if isinstance(row, dict)
            and row.get("name")
            and str(row.get("declaration", "")).strip()
        }
    raise SystemExit(f"unsupported intrinsic declaration JSON shape: {path}")


VECTOR_TYPE_RE = re.compile(r"^v(?P<lanes>\d+)(?:x)?(?P<elem>[A-Za-z0-9]+)$")
VECTOR_OP_HINT_RE = re.compile(r"^(?:ir|llvm)\.vector\.(?P<opcode>[a-z0-9_]+)\.v(?P<lanes>\d+)x(?P<elem>.+)$")
LLVM_INTRINSIC_ALIASES = {
    "_tile_dpbssd": "llvm.x86.tdpbssd.internal",
    "_tile_dpbusd": "llvm.x86.tdpbusd.internal",
    "amx.tile.tdpbssd": "llvm.x86.tdpbssd.internal",
    "amx.tile.tdpbusd": "llvm.x86.tdpbusd.internal",
}


def normalize_hint_name(text: str) -> str:
    text = LLVM_INTRINSIC_ALIASES.get(text, text)
    if text.startswith("llvm.vector.") and not text.startswith("llvm.vector.reduce."):
        return "ir.vector." + text.removeprefix("llvm.vector.")
    return text


def hint_text(item: dict[str, Any]) -> str:
    intrinsic = str(item.get("intrinsic") or "").strip()
    label = str(item.get("label") or "").strip()
    text = intrinsic or label
    return normalize_hint_name(text)


def vector_hint_key(text: str) -> tuple[str, str] | None:
    parts = text.split(".")
    vector_indices = [index for index, part in enumerate(parts) if VECTOR_TYPE_RE.fullmatch(part)]
    if not vector_indices:
        return None
    index = vector_indices[-1]
    vector = VECTOR_TYPE_RE.fullmatch(parts[index])
    assert vector is not None
    elem = vector.group("elem")
    base = ".".join(parts[:index])
    suffix = ".".join(parts[index + 1 :])
    return (f"{base}.*{suffix}", elem)


def vector_hint_width(text: str) -> int:
    widths = [
        int(match.group("lanes"))
        for part in text.split(".")
        if (match := VECTOR_TYPE_RE.fullmatch(part))
    ]
    return max(widths) if widths else 0


def vector_op_ir_syntax(text: str) -> str | None:
    match = VECTOR_OP_HINT_RE.fullmatch(text)
    if not match:
        return None
    if text.startswith("llvm.vector.reduce."):
        return None
    return f"{match.group('opcode')} <{match.group('lanes')} x {match.group('elem')}>"


def compact_hint_texts(rows: list[dict[str, Any]], max_regions: int, top_k: int) -> list[str]:
    selected_rows = sorted(
        rows,
        key=lambda row: float(row.get("change_probability", 0.0)),
        reverse=True,
    )[:max_regions]
    best_by_vector_key: dict[tuple[str, str], str] = {}
    scalar_hints: list[str] = []
    seen_scalar: set[str] = set()
    for row in selected_rows:
        for item in row.get("recommendations", [])[:top_k]:
            text = hint_text(item)
            if not text:
                continue
            key = vector_hint_key(text)
            if key is None:
                if text not in seen_scalar:
                    scalar_hints.append(text)
                    seen_scalar.add(text)
                continue
            current = best_by_vector_key.get(key)
            if current is None or vector_hint_width(text) > vector_hint_width(current):
                best_by_vector_key[key] = text
    compacted: list[str] = []
    seen: set[str] = set()
    for text in [*scalar_hints, *best_by_vector_key.values()]:
        if text in seen:
            continue
        compacted.append(text)
        seen.add(text)
    return compacted


def format_recommendations(
    rows: list[dict[str, Any]],
    max_regions: int,
    top_k: int,
    intrinsic_declarations: dict[str, str],
) -> str:
    hints = compact_hint_texts(rows, max_regions, top_k)
    if not hints:
        return ""
    vector_ops = [(hint, syntax) for hint in hints if (syntax := vector_op_ir_syntax(hint))]
    intrinsic_hints = [
        (hint, clean_intrinsic_declaration(intrinsic_declarations.get(hint, "")))
        for hint in hints
        if not vector_op_ir_syntax(hint)
    ]
    has_declarations = any(declaration for _hint, declaration in intrinsic_hints)
    lines = [
        "Retriever-proposed vectorization hints:",
        "- Treat these as optional optimization hints, not mandatory rewrites.",
        "- Prefer a hint only when it is semantically valid for the IR and profitable for the target.",
    ]
    if has_declarations:
        lines.append(
            "- Exact declarations are generated from LLVM; if you use one of these intrinsics, "
            "match its declaration and call operand order exactly."
        )
    if vector_ops:
        lines.append("- Ordinary LLVM IR vector operations:")
        for _hint, syntax in vector_ops:
            lines.append(f"  - {syntax}")
    if intrinsic_hints:
        lines.append("- LLVM intrinsic hints:")
    for hint, declaration in intrinsic_hints:
        if declaration:
            lines.append(f"  - {declaration}")
        else:
            lines.append(f"  - {hint}")
    return "\n".join(lines)


def render_prompt(
    template: str,
    *,
    mode: str,
    task: dict[str, Any],
    source_ir: str,
    recommendation_block: str,
    target: str,
    target_cpu: str,
    target_cpu_name: str,
    target_notes: str,
) -> str:
    return template.format(
        kernel=task["kernel"],
        mode=mode,
        target=target,
        target_cpu=target_cpu,
        target_cpu_name=target_cpu_name,
        target_notes=target_notes,
        recommendation_block=recommendation_block,
        source_ir=source_ir,
    )


def render_repair_prompt(
    template: str,
    *,
    mode: str,
    task: dict[str, Any],
    source_ir: str,
    candidate_ir: str,
    verifier_result: dict[str, Any],
    iteration: int,
    max_iterations: int,
    target: str,
    target_cpu: str,
    target_cpu_name: str,
    target_notes: str,
) -> str:
    return template.format(
        kernel=task["kernel"],
        mode=mode,
        target=target,
        target_cpu=target_cpu,
        target_cpu_name=target_cpu_name,
        target_notes=target_notes,
        iteration=iteration + 1,
        max_iterations=max_iterations,
        source_ir=source_ir,
        candidate_ir=candidate_ir,
        verifier_result=json.dumps(verifier_result, indent=2, sort_keys=True),
    )


def iteration_dir(mode_dir: Path, iteration: int) -> Path:
    return mode_dir if iteration == 0 else mode_dir / f"iter_{iteration:02d}"


def call_llm(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    if args.agent_cmd:
        return run_cmd(shlex.split(args.agent_cmd, posix=os.name != "nt"), timeout=args.agent_timeout, input_text=prompt)
    start = time.time()
    try:
        response = args._llm_client.call(
            prompt_text=prompt,
            model=args.model,
            max_output_tokens=args.max_tokens,
            temperature=args.temperature,
            truncation=args.truncation,
            store=args.store,
            api_mode=args.api_mode,
            max_retries=args.max_retries,
            base_backoff=args.base_backoff,
        )
        return {
            "returncode": 0,
            "stdout": response,
            "stderr": "",
            "elapsed_sec": round(time.time() - start, 3),
            "ok": True,
            "model": args.model,
            "api_mode": args.api_mode,
        }
    except Exception as exc:  # noqa: BLE001 - experiment runner records failures.
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": str(exc),
            "elapsed_sec": round(time.time() - start, 3),
            "ok": False,
            "model": args.model,
            "api_mode": args.api_mode,
        }


def run_mode(
    args: argparse.Namespace,
    *,
    mode: str,
    task: dict[str, Any],
    task_dir: Path,
    template: str,
    repair_template: str,
    seed_mode_dir: Path | None,
    scalar_ir_path: Path,
    scalar_ir: str,
    recommendation_rows: list[dict[str, Any]],
    expected_checksum: float | None,
    reference_time: float | None,
    scalar_time: float | None,
) -> dict[str, Any]:
    mode_dir = task_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    recommendation_block = (
        ""
        if mode == "baseline"
        else format_recommendations(
            recommendation_rows,
            args.max_recommendation_regions,
            args.recommendation_top_k,
            args.intrinsic_declarations,
        )
    )
    prompt = render_prompt(
        template,
        mode=mode,
        task=task,
        source_ir=scalar_ir,
        recommendation_block=recommendation_block,
        target=args.target,
        target_cpu=args.target_cpu,
        target_cpu_name=args.target_cpu_name,
        target_notes=args.target_notes,
    )
    record: dict[str, Any] = {
        "mode": mode,
        "recommendation_regions": len(recommendation_rows) if mode != "baseline" else 0,
        "max_iterations": args.max_iters,
        "iterations": [],
    }
    if args.prompt_only:
        prompt_path = mode_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        record["prompt_path"] = str(prompt_path)
        record["ok"] = False
        record["prompt_only"] = True
        return record
    if expected_checksum is None and not args.generate_only:
        record["ok"] = False
        record["error"] = "verification requires expected checksum"
        return record

    current_prompt = prompt
    previous_candidate = ""
    previous_verify: dict[str, Any] = {}
    for iteration in range(args.max_iters):
        attempt_dir = iteration_dir(mode_dir, iteration)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        if iteration > 0:
            current_prompt = render_repair_prompt(
                repair_template,
                mode=mode,
                task=task,
                source_ir=scalar_ir,
                candidate_ir=previous_candidate,
                verifier_result=repair_verifier_feedback(previous_verify),
                iteration=iteration,
                max_iterations=args.max_iters,
                target=args.target,
                target_cpu=args.target_cpu,
                target_cpu_name=args.target_cpu_name,
                target_notes=args.target_notes,
            )

        prompt_path = attempt_dir / "prompt.txt"
        llm_path = attempt_dir / "llm.json"
        candidate_ir_path = attempt_dir / "candidate.ll"
        postprocess_path = attempt_dir / "postprocess.json"
        verify_path = attempt_dir / "verify.json"
        prompt_path.write_text(current_prompt, encoding="utf-8")
        existing_dir = seed_mode_dir if iteration == 0 and seed_mode_dir is not None else attempt_dir
        existing_llm_path = existing_dir / "llm.json"
        existing_candidate_path = existing_dir / "candidate.ll"
        existing_llm_result = (
            successful_existing_generation(existing_llm_path, existing_candidate_path)
            if args.skip_existing
            else None
        )
        if existing_llm_result is not None:
            llm_result = existing_llm_result
            if existing_llm_path.resolve() != llm_path.resolve():
                llm_path.write_text(json.dumps(llm_result, indent=2), encoding="utf-8")
                shutil.copyfile(existing_candidate_path, candidate_ir_path)
        else:
            llm_result = call_llm(args, current_prompt)
            llm_path.write_text(json.dumps(llm_result, indent=2), encoding="utf-8")

        attempt: dict[str, Any] = {
            "iteration": iteration,
            "prompt_path": str(prompt_path),
            "llm_path": str(llm_path),
            "candidate_ir": str(candidate_ir_path),
            "postprocess_path": str(postprocess_path),
            "verify_path": str(verify_path),
            "llm_ok": bool(llm_result["ok"]),
            "llm_elapsed_sec": llm_result.get("elapsed_sec"),
            "skipped_existing_generation": existing_llm_result is not None,
            "candidate_origin": (
                "seed"
                if existing_llm_result is not None and iteration == 0 and seed_mode_dir is not None
                else ("existing" if existing_llm_result is not None else "generated")
            ),
        }
        record["iterations"].append(attempt)
        if not llm_result["ok"]:
            attempt.update(
                {
                    "ok": False,
                    "error": llm_result.get("error") or llm_result.get("stderr"),
                    "alive2_ok": None,
                    "differential_fuzz_ok": None,
                    "compile_ok": None,
                    "run_ok": None,
                    "checksum_ok": None,
                }
            )
            record.update(attempt)
            record["ok"] = False
            record["error"] = attempt["error"]
            break

        if existing_llm_result is None:
            candidate_ir = strip_markdown_fence(str(llm_result["stdout"]))
            candidate_ir_path.write_text(candidate_ir, encoding="utf-8")
        postprocess_summary = postprocess_ir_if_enabled(args, candidate_ir_path)
        postprocess_path.write_text(json.dumps(postprocess_summary, indent=2), encoding="utf-8")
        module_target_sync = synchronize_module_target(scalar_ir, candidate_ir_path)
        attr_patch = patch_target_attributes_if_enabled(args, candidate_ir_path)
        if existing_llm_result is not None:
            attr_patch["skipped_existing_generation"] = True
        attempt["ir_postprocess"] = postprocess_summary
        attempt["module_target_sync"] = module_target_sync
        attempt["target_attribute_patch"] = attr_patch

        if args.generate_only:
            attempt.update({"ok": True, "generated_only": True, "verify_skipped": True})
            record.update(attempt)
            record["ok"] = True
            record["generated_only"] = True
            break

        verify_result = validate_ir_candidate(
            args,
            task,
            scalar_ir_path,
            candidate_ir_path,
            attempt_dir / "candidate",
            float(expected_checksum),
        )
        verify_path.write_text(json.dumps(verify_result, indent=2), encoding="utf-8")
        differential_result = verify_result["differential"]
        parsed = differential_result.get("parsed") or {}
        observed_time = parsed.get("time_sec")
        attempt.update(
            {
                "ok": verify_result["ok"],
                "repairable": verify_result["repairable"],
                "semantic_statuses": verify_result["semantic_statuses"],
                "alive2_ok": verify_result["alive2"]["ok"],
                "alive2_status": verify_result["alive2"]["status"],
                "differential_fuzz_ok": verify_result["differential_fuzz"]["ok"],
                "differential_fuzz_status": verify_result["differential_fuzz"]["status"],
                "tsvc_ok": differential_result["ok"],
                "compile_ok": differential_result["compile"]["ok"],
                "run_ok": bool(differential_result.get("run") and differential_result["run"]["ok"]),
                "checksum_ok": differential_result["checksum_ok"],
                "observed_checksum": parsed.get("checksum"),
                "observed_time_sec": observed_time,
                "reference_time_sec": reference_time,
                "scalar_ir_time_sec": scalar_time,
                "speedup_vs_scalar_ir": (
                    float(scalar_time) / float(observed_time)
                    if scalar_time and observed_time and float(observed_time) > 0.0
                    else None
                ),
                "speedup_vs_reference": (
                    float(reference_time) / float(observed_time)
                    if reference_time and observed_time and float(observed_time) > 0.0
                    else None
                ),
            }
        )
        record.update(attempt)
        record["ok"] = bool(attempt["ok"])
        if attempt["ok"]:
            record["success_iteration"] = iteration
            break
        if not verify_result["repairable"]:
            record["verifier_inconclusive"] = True
            record["error"] = "semantic validators were inconclusive; candidate repair was not attempted"
            break
        previous_candidate = candidate_ir_path.read_text(encoding="utf-8", errors="replace")
        previous_verify = verify_result
    return record


def parse_modes(text: str) -> list[str]:
    modes = [item.strip() for item in text.split(",") if item.strip()]
    allowed = {"baseline", "guided"}
    bad = sorted(set(modes).difference(allowed))
    if bad:
        raise SystemExit(f"unknown modes: {', '.join(bad)}")
    return modes or ["baseline", "guided"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline LLVM-IR LLM optimization against recommendation-guided TSVC IR optimization."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--recommendations-jsonl", type=Path)
    parser.add_argument(
        "--intrinsic-declarations-json",
        type=Path,
        help="JSON/JSONL map produced by resolve_llvm_intrinsic_declarations.py.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--seed-results-dir",
        type=Path,
        help="Read attempt-zero llm.json and candidate.ll from another result directory with --skip-existing.",
    )
    parser.add_argument("--agent-cmd", default="", help="Compatibility path: command that reads prompt stdin and writes IR stdout.")
    parser.add_argument("--llm-config", "--config", dest="llm_config", type=Path, help="YAML/JSON config with optional llm: section.")
    parser.add_argument("--model", default="", help="Override config llm.llm_model / OPENAI_MODEL.")
    parser.add_argument("--api-mode", default="", choices=["", "auto", "responses", "chat"])
    parser.add_argument("--api-base-url", default="", help="Override config llm.base_url / OPENAI_BASE_URL.")
    parser.add_argument("--api-key", default="", help="Override config llm.api_key / OPENAI_API_KEY.")
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--truncation", default="auto")
    parser.add_argument("--store", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--base-backoff", type=float, default=1.0)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--repair-template", type=Path, default=DEFAULT_REPAIR_TEMPLATE)
    parser.add_argument(
        "--max-iters",
        type=int,
        default=5,
        help="Maximum attempts per mode, including generation.",
    )
    parser.add_argument("--tsvc-root", type=Path)
    parser.add_argument("--ir-dir", type=Path, default=REPO_ROOT / "runs/tsvc_prevec_ir")
    parser.add_argument("--kernels", default="all")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--modes", default="baseline,guided")
    parser.add_argument("--target", default="x86_64-pc-linux-gnu")
    parser.add_argument(
        "--target-cpu",
        default="auto",
        help="LLVM target CPU to describe in prompts; 'auto' detects clang -march=native.",
    )
    parser.add_argument(
        "--target-cpu-name",
        default="",
        help="Human-readable CPU model name for prompts; auto-detected when omitted.",
    )
    parser.add_argument("--target-detect-timeout", type=int, default=10)
    parser.add_argument("--target-notes", default="System V AMD64 ABI, optimize for native x86 vector execution.")
    parser.add_argument(
        "--target-attr-march",
        default="native",
        help="Architecture passed to clang -march when detecting and emitting target attributes.",
    )
    parser.add_argument(
        "--no-target-attributes",
        action="store_true",
        help="Do not inject target-cpu/target-features attributes into scalar or generated IR.",
    )
    parser.add_argument(
        "--no-ir-postprocess",
        action="store_true",
        help="Disable text post-processing of LLM-generated candidate IR before verification.",
    )
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--scalar-opt-level", default="3", choices=["0", "1", "2", "3", "s", "z"])
    parser.add_argument("--extra-clang-arg", action="append", default=[])
    parser.add_argument("--skip-existing-ir", action="store_true")
    parser.add_argument(
        "--skip-scalar-ir-eval",
        action="store_true",
        help="Skip compiling/running the scalar IR performance baseline during verification.",
    )
    parser.add_argument("--max-recommendation-regions", type=int, default=8)
    parser.add_argument("--recommendation-top-k", type=int, default=20)
    parser.add_argument("--checksum-atol", type=float, default=1e-3)
    parser.add_argument("--build-timeout", type=int, default=60)
    parser.add_argument("--run-timeout", type=int, default=300)
    parser.add_argument(
        "--alive2",
        action="store_true",
        help="Require Alive2 refinement verification before accepting IR.",
    )
    parser.add_argument("--alive-tv", default="alive-tv", help="Path to the Alive2 alive-tv executable.")
    parser.add_argument("--alive2-arg", action="append", default=[], help="Additional argument passed to alive-tv.")
    parser.add_argument("--alive2-timeout", type=int, default=300)
    parser.add_argument(
        "--differential-fuzz",
        action="store_true",
        help="Require merged reference/candidate IR differential testing with libFuzzer.",
    )
    parser.add_argument("--llvm-link", default="llvm-link")
    parser.add_argument("--fuzz-runs", type=int, default=1000)
    parser.add_argument("--fuzz-max-len", type=int, default=4096)
    parser.add_argument("--fuzz-input-timeout", type=int, default=5)
    parser.add_argument("--fuzz-timeout", type=int, default=900)
    parser.add_argument("--fuzz-sanitizers", default="fuzzer,address")
    parser.add_argument("--agent-timeout", type=int, default=900)
    parser.add_argument("--prompt-only", action="store_true")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Call the LLM and write candidate IR, but skip candidate compile/run verification.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing successful llm.json + candidate.ll outputs instead of calling the LLM again.",
    )
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help="Reuse successful mode records from out-dir/records.jsonl without rerunning their validators.",
    )
    parser.add_argument(
        "--revalidate-kernels",
        default="",
        help="Comma-separated kernels whose successful records must be revalidated during resume.",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Number of mode jobs to run concurrently.")
    args = parser.parse_args()

    if args.limit < 0:
        args.limit = 0
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    if args.max_iters <= 0:
        parser.error("--max-iters must be positive")
    if args.fuzz_runs <= 0:
        parser.error("--fuzz-runs must be positive")
    if args.fuzz_max_len <= 0:
        parser.error("--fuzz-max-len must be positive")
    if args.fuzz_input_timeout <= 0 or args.fuzz_timeout <= 0:
        parser.error("fuzz timeouts must be positive")
    if args.seed_results_dir and not args.skip_existing:
        parser.error("--seed-results-dir requires --skip-existing")
    if args.resume_completed and not args.skip_existing:
        parser.error("--resume-completed requires --skip-existing")
    if args.prompt_only and args.generate_only:
        parser.error("--prompt-only and --generate-only are mutually exclusive")
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if args.max_recommendation_regions <= 0:
        parser.error("--max-recommendation-regions must be positive")
    if args.recommendation_top_k <= 0:
        parser.error("--recommendation-top-k must be positive")

    if not args.prompt_only and not args.agent_cmd:
        from neural_asm_pipeline.llm_client import client_from_config, config_value

        client, section = client_from_config(
            args.llm_config,
            base_url=args.api_base_url,
            api_key=args.api_key,
        )
        args._llm_client = client
        args.model = args.model or str(config_value(section, "llm_model", "OPENAI_MODEL", "gpt-5"))
        args.api_mode = args.api_mode or str(config_value(section, "api_mode", "OPENAI_API_MODE", "auto"))
    elif args.agent_cmd:
        args.model = args.model or "agent-cmd"
        args.api_mode = args.api_mode or "agent-cmd"

    from neural_asm_pipeline.target_detect import detect_target

    target_info = detect_target(
        compiler=args.clang,
        target=args.target,
        target_cpu=args.target_cpu,
        target_cpu_name=args.target_cpu_name,
        timeout=args.target_detect_timeout,
    )
    args.target_cpu = str(target_info["target_cpu"])
    args.target_cpu_name = str(target_info["target_cpu_name"])
    target_attr_info: dict[str, Any] = {"attributes": {}, "disabled": args.no_target_attributes}
    if not args.no_target_attributes:
        from neural_asm_pipeline.target_attributes import target_attribute_values

        target_attr_info = target_attribute_values(
            clang=args.clang,
            target=args.target,
            target_cpu=args.target_cpu,
            fallback_features=target_info.get("target_features", []),
            march=args.target_attr_march,
            timeout=args.target_detect_timeout,
        )
    args.target_attributes = dict(target_attr_info.get("attributes", {}))

    modes = parse_modes(args.modes)
    revalidate_kernels = {item.strip() for item in args.revalidate_kernels.split(",") if item.strip()}
    template = args.template.read_text(encoding="utf-8")
    repair_template = args.repair_template.read_text(encoding="utf-8")
    recommendations = load_recommendations(args.recommendations_jsonl)
    args.intrinsic_declarations = load_intrinsic_declarations(args.intrinsic_declarations_json)
    selected = select_tasks(load_jsonl(args.manifest), args.kernels, args.offset, args.limit)
    if not selected:
        parser.error("selected task range is empty")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_info["target_attributes"] = target_attr_info
    (args.out_dir / "target.json").write_text(json.dumps(target_info, indent=2), encoding="utf-8")
    records_path = args.out_dir / "records.jsonl"
    resume_snapshot_path = args.out_dir / "records.resume_snapshot.jsonl"
    previous_by_task: dict[tuple[int, str], dict[str, Any]] = {}
    if args.resume_completed:
        resume_sources = [path for path in (resume_snapshot_path, records_path) if path.exists()]
        for source in resume_sources:
            for record in load_jsonl(source):
                if "task_index" in record and "kernel" in record:
                    previous_by_task[(int(record["task_index"]), str(record["kernel"]))] = record
        if previous_by_task and not resume_snapshot_path.exists():
            write_jsonl(resume_snapshot_path, list(previous_by_task.values()))
    records_path.write_text("", encoding="utf-8")
    records: list[dict[str, Any]] = []
    work_items: list[dict[str, Any]] = []
    needs_verification = not args.prompt_only and not args.generate_only

    for task_index, task in selected:
        kernel = str(task["kernel"])
        task_dir = args.out_dir / f"{task_index:04d}_{kernel}"
        task_dir.mkdir(parents=True, exist_ok=True)
        scalar_ir_path = args.ir_dir / f"{kernel}.ll"
        previous_record = previous_by_task.get((task_index, kernel), {})
        previous_modes = previous_record.get("modes", {}) if isinstance(previous_record, dict) else {}
        force_revalidate = kernel in revalidate_kernels
        if revalidate_kernels and not force_revalidate and previous_record:
            records.append(previous_record)
            write_jsonl(records_path, records)
            print(
                f"[{task_index}/{len(kernels)}] {kernel}: preserved existing record "
                "(not selected for revalidation)",
                flush=True,
            )
            continue
        if not force_revalidate and previous_modes and all(previous_modes.get(mode, {}).get("ok") for mode in modes):
            records.append(previous_record)
            write_jsonl(records_path, records)
            print(
                json.dumps(
                    {
                        "task_index": task_index,
                        "kernel": kernel,
                        "ok": True,
                        "resume_skipped_completed": True,
                        "modes": modes,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue

        ir_result = compile_scalar_ir(args, task, scalar_ir_path)
        (task_dir / "scalar_ir_emit.json").write_text(json.dumps(ir_result, indent=2), encoding="utf-8")
        if not ir_result["ok"]:
            record = {
                "task_index": task_index,
                "kernel": kernel,
                "ok": False,
                "error": "scalar IR emit failed",
                "scalar_ir_emit": ir_result,
            }
            records.append(record)
            write_jsonl(records_path, records)
            print(json.dumps({"task_index": task_index, "kernel": kernel, "ok": False, "error": record["error"]}), flush=True)
            continue
        scalar_attr_patch = patch_target_attributes_if_enabled(args, scalar_ir_path)
        (task_dir / "scalar_ir_target_attrs.json").write_text(
            json.dumps(scalar_attr_patch, indent=2),
            encoding="utf-8",
        )

        expected_checksum: float | None = None
        reference_time: float | None = None
        scalar_time: float | None = None
        if needs_verification:
            reference = build_reference(args, task, task_dir / "reference")
            (task_dir / "reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
            if not reference["ok"]:
                record = {
                    "task_index": task_index,
                    "kernel": kernel,
                    "ok": False,
                    "error": "reference build/run failed",
                    "reference": reference,
                }
                records.append(record)
                write_jsonl(records_path, records)
                print(
                    json.dumps({"task_index": task_index, "kernel": kernel, "ok": False, "error": record["error"]}),
                    flush=True,
                )
                continue

            expected_checksum = float(reference["parsed"]["checksum"])
            reference_time = float(reference["parsed"]["time_sec"])
            if args.skip_scalar_ir_eval:
                scalar_eval = {
                    "ok": False,
                    "skipped": True,
                    "reason": "--skip-scalar-ir-eval",
                    "ir_file": str(scalar_ir_path),
                    "expected_checksum": expected_checksum,
                }
            else:
                scalar_eval = build_and_run_ir(args, task, scalar_ir_path, task_dir / "scalar_ir", expected_checksum)
            (task_dir / "scalar_ir.json").write_text(json.dumps(scalar_eval, indent=2), encoding="utf-8")
            scalar_parsed = scalar_eval.get("parsed") or {}
            scalar_time = float(scalar_parsed["time_sec"]) if scalar_eval["ok"] and scalar_parsed.get("time_sec") else None
        scalar_ir = scalar_ir_path.read_text(encoding="utf-8", errors="replace")
        recommendation_rows = recommendations_for_task(recommendations, task, kernel)

        record = {
            "task_index": task_index,
            "kernel": kernel,
            "scalar_ir": str(scalar_ir_path),
            "reference_time_sec": reference_time,
            "scalar_ir_time_sec": scalar_time,
            "scalar_ir_target_attribute_patch": scalar_attr_patch,
            "expected_checksum": expected_checksum,
            "recommendation_regions": len(recommendation_rows),
            "modes": {
                mode: previous_modes[mode]
                for mode in modes
                if not force_revalidate and previous_modes.get(mode, {}).get("ok")
            },
            "ok": not force_revalidate and any(previous_modes.get(mode, {}).get("ok") for mode in modes),
        }
        records.append(record)
        for mode in modes:
            if record["modes"].get(mode, {}).get("ok"):
                continue
            work_items.append(
                {
                    "record": record,
                    "task_index": task_index,
                    "kernel": kernel,
                    "mode": mode,
                    "task": task,
                    "task_dir": task_dir,
                    "template": template,
                    "seed_mode_dir": (
                        args.seed_results_dir / task_dir.name / mode if args.seed_results_dir else None
                    ),
                    "scalar_ir_path": scalar_ir_path,
                    "scalar_ir": scalar_ir,
                    "recommendation_rows": recommendation_rows,
                    "expected_checksum": expected_checksum,
                    "reference_time": reference_time,
                    "scalar_time": scalar_time,
                }
            )
        write_jsonl(records_path, records)

    if work_items:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_item = {
                executor.submit(
                    run_mode,
                    args,
                    mode=item["mode"],
                    task=item["task"],
                    task_dir=item["task_dir"],
                    template=item["template"],
                    repair_template=repair_template,
                    seed_mode_dir=item["seed_mode_dir"],
                    scalar_ir_path=item["scalar_ir_path"],
                    scalar_ir=item["scalar_ir"],
                    recommendation_rows=item["recommendation_rows"],
                    expected_checksum=item["expected_checksum"],
                    reference_time=item["reference_time"],
                    scalar_time=item["scalar_time"],
                ): item
                for item in work_items
            }
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                mode = str(item["mode"])
                record = item["record"]
                try:
                    mode_record = future.result()
                except Exception as exc:  # noqa: BLE001 - keep batch generation moving.
                    mode_record = {
                        "mode": mode,
                        "ok": False,
                        "error": str(exc),
                        "exception_type": type(exc).__name__,
                    }
                record["modes"][mode] = mode_record
                record["ok"] = any(mode_record.get("ok") for mode_record in record["modes"].values())
                write_jsonl(records_path, records)
                print(
                    json.dumps(
                        {
                            "task_index": item["task_index"],
                            "kernel": item["kernel"],
                            "mode": mode,
                            "ok": mode_record.get("ok", False),
                            "time_sec": mode_record.get("observed_time_sec"),
                            "speedup_vs_scalar_ir": mode_record.get("speedup_vs_scalar_ir"),
                            "generated_only": mode_record.get("generated_only", False),
                            "skipped_existing_generation": mode_record.get("skipped_existing_generation", False),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    summary: dict[str, Any] = {
        "tasks": len(records),
        "modes": modes,
        "target": args.target,
        "target_cpu": args.target_cpu,
        "target_cpu_name": args.target_cpu_name,
        "target_info": str(args.out_dir / "target.json"),
        "intrinsic_declarations": (
            str(args.intrinsic_declarations_json) if args.intrinsic_declarations_json else None
        ),
        "intrinsic_declaration_count": len(args.intrinsic_declarations),
        "prompt_only": args.prompt_only,
        "generate_only": args.generate_only,
        "skip_existing": args.skip_existing,
        "resume_completed": args.resume_completed,
        "revalidate_kernels": sorted(revalidate_kernels),
        "seed_results_dir": str(args.seed_results_dir) if args.seed_results_dir else None,
        "ir_postprocess": not args.no_ir_postprocess,
        "skip_scalar_ir_eval": args.skip_scalar_ir_eval,
        "jobs": args.jobs,
        "max_iters": args.max_iters,
        "validation_methods": (
            (["alive2"] if args.alive2 else [])
            + ["reference-oracle-checksum-differential"]
            + (["libfuzzer-merged-ir-differential"] if args.differential_fuzz else [])
        ),
        "acceptance_policy": (
            "tsvc-pass-and-semantic-pass-without-counterexample"
            if args.alive2 or args.differential_fuzz
            else "tsvc-checksum"
        ),
        "differential_fuzz": {
            "enabled": args.differential_fuzz,
            "llvm_link": args.llvm_link if args.differential_fuzz else None,
            "runs": args.fuzz_runs if args.differential_fuzz else None,
            "max_len": args.fuzz_max_len if args.differential_fuzz else None,
            "input_timeout": args.fuzz_input_timeout if args.differential_fuzz else None,
            "timeout": args.fuzz_timeout if args.differential_fuzz else None,
            "sanitizers": args.fuzz_sanitizers if args.differential_fuzz else None,
        },
        "records": str(records_path),
        "successes": {
            mode: sum(1 for record in records if record.get("modes", {}).get(mode, {}).get("ok"))
            for mode in modes
        },
        "pass_at_1": {
            mode: sum(
                1
                for record in records
                if record.get("modes", {}).get(mode, {}).get("iterations")
                and record["modes"][mode]["iterations"][0].get("ok")
            )
            for mode in modes
        },
        "recovered_by_self_correction": {
            mode: sum(
                1
                for record in records
                if record.get("modes", {}).get(mode, {}).get("ok")
                and int(record.get("modes", {}).get(mode, {}).get("success_iteration", 0)) > 0
            )
            for mode in modes
        },
        "attempt_zero_validation": {
            mode: {
                "seeded": sum(
                    1
                    for record in records
                    if record.get("modes", {}).get(mode, {}).get("iterations")
                    and record["modes"][mode]["iterations"][0].get("candidate_origin") == "seed"
                ),
                "tsvc_pass": sum(
                    1
                    for record in records
                    if record.get("modes", {}).get(mode, {}).get("iterations")
                    and record["modes"][mode]["iterations"][0].get("tsvc_ok")
                ),
                "alive2_pass": sum(
                    1
                    for record in records
                    if record.get("modes", {}).get(mode, {}).get("iterations")
                    and record["modes"][mode]["iterations"][0].get("alive2_status") == "pass"
                ),
                "differential_fuzz_pass": sum(
                    1
                    for record in records
                    if record.get("modes", {}).get(mode, {}).get("iterations")
                    and record["modes"][mode]["iterations"][0].get("differential_fuzz_status") == "pass"
                ),
                "accepted": sum(
                    1
                    for record in records
                    if record.get("modes", {}).get(mode, {}).get("iterations")
                    and record["modes"][mode]["iterations"][0].get("ok")
                ),
            }
            for mode in modes
        },
    }
    for mode in modes:
        times = [
            float(record["modes"][mode]["observed_time_sec"])
            for record in records
            if record.get("modes", {}).get(mode, {}).get("ok")
            and record["modes"][mode].get("observed_time_sec") is not None
        ]
        summary[f"{mode}_mean_time_sec"] = sum(times) / len(times) if times else None
    if "baseline" in modes and "guided" in modes:
        paired_speedups: list[float] = []
        for record in records:
            baseline = record.get("modes", {}).get("baseline", {})
            guided = record.get("modes", {}).get("guided", {})
            if (
                baseline.get("ok")
                and guided.get("ok")
                and baseline.get("observed_time_sec") is not None
                and guided.get("observed_time_sec") is not None
            ):
                baseline_time = float(baseline["observed_time_sec"])
                guided_time = float(guided["observed_time_sec"])
                if guided_time > 0.0:
                    paired_speedups.append(baseline_time / guided_time)
        summary["guided_speedup_vs_llm_baseline_mean"] = (
            sum(paired_speedups) / len(paired_speedups) if paired_speedups else None
        )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
