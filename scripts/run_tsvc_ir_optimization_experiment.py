#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
DEFAULT_TEMPLATE = REPO_ROOT / "prompts/llvm_ir_vectorize_fast_v1.txt"


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
        return {
            "cmd": cmd,
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "elapsed_sec": round(time.time() - start, 3),
            "ok": False,
            "error": "timeout",
        }


def parse_tsvc_output(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return {"ok": False, "error": "empty stdout"}
    data_line = lines[-1]
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", data_line)
    if len(numbers) < 2:
        return {"ok": False, "error": f"could not parse TSVC line: {data_line!r}"}
    try:
        return {
            "ok": True,
            "line": data_line,
            "time_sec": float(numbers[-2]),
            "checksum": float(numbers[-1]),
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
    paths = task_paths(args, task)
    ir_path.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_existing_ir and ir_path.exists():
        return {"ok": True, "skipped": True, "ir_file": str(ir_path)}
    cmd = [
        args.clang,
        "-S",
        "-emit-llvm",
        f"-O{args.scalar_opt_level}",
        "-fno-vectorize",
        "-fno-slp-vectorize",
        "-fno-unroll-loops",
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
        "-march=native",
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
        "-march=native",
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
        "exe": str(exe),
        "compile": compile_result,
        "run": run_result,
        "parsed": parsed,
        "expected_checksum": expected_checksum,
        "checksum_atol": args.checksum_atol,
        "checksum_ok": checksum_ok,
        "ok": bool(compile_result["ok"] and run_result and run_result["ok"] and parsed and parsed["ok"] and checksum_ok),
    }


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


def format_recommendations(rows: list[dict[str, Any]], max_regions: int, top_k: int) -> str:
    if not rows:
        return ""
    selected = sorted(
        rows,
        key=lambda row: float(row.get("change_probability", 0.0)),
        reverse=True,
    )[:max_regions]
    lines = [
        "Retriever-proposed vectorization hints:",
        "- Treat these as optional optimization hints, not mandatory rewrites.",
        "- Prefer a hint only when it is semantically valid for the IR and profitable for the target.",
    ]
    for row in selected:
        lines.append(
            "- region {region_index}: change_probability={change_probability:.4f}, nodes={nodes}, edges={edges}".format(
                region_index=row.get("region_index"),
                change_probability=float(row.get("change_probability", 0.0)),
                nodes=row.get("num_nodes"),
                edges=row.get("num_edges"),
            )
        )
        tokens = row.get("node_tokens") or []
        if tokens:
            lines.append("  region_tokens: " + " ".join(str(token) for token in tokens[:48]))
        for item in row.get("recommendations", [])[:top_k]:
            parts = [
                f"rank={item.get('rank')}",
                f"label={item.get('label')}",
                f"score={float(item.get('score', 0.0)):.4f}",
            ]
            if item.get("intrinsic"):
                parts.append(f"intrinsic={item.get('intrinsic')}")
            if item.get("instruction"):
                parts.append(f"instruction={item.get('instruction')}")
            lines.append("  - " + ", ".join(parts))
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


def call_llm(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    if args.agent_cmd:
        return run_cmd(shlex.split(args.agent_cmd), timeout=args.agent_timeout, input_text=prompt)
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
    scalar_ir: str,
    recommendation_rows: list[dict[str, Any]],
    expected_checksum: float,
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
    prompt_path = mode_dir / "prompt.txt"
    llm_path = mode_dir / "llm.json"
    candidate_ir_path = mode_dir / "candidate.ll"
    verify_path = mode_dir / "verify.json"
    prompt_path.write_text(prompt, encoding="utf-8")

    record: dict[str, Any] = {
        "mode": mode,
        "prompt_path": str(prompt_path),
        "llm_path": str(llm_path),
        "candidate_ir": str(candidate_ir_path),
        "verify_path": str(verify_path),
        "recommendation_regions": len(recommendation_rows) if mode != "baseline" else 0,
    }
    if args.prompt_only:
        record["ok"] = False
        record["prompt_only"] = True
        return record

    llm_result = call_llm(args, prompt)
    llm_path.write_text(json.dumps(llm_result, indent=2), encoding="utf-8")
    record["llm_ok"] = llm_result["ok"]
    record["llm_elapsed_sec"] = llm_result["elapsed_sec"]
    if not llm_result["ok"]:
        record["ok"] = False
        record["error"] = llm_result.get("error") or llm_result.get("stderr")
        return record

    candidate_ir = strip_markdown_fence(str(llm_result["stdout"]))
    candidate_ir_path.write_text(candidate_ir, encoding="utf-8")
    verify_result = build_and_run_ir(args, task, candidate_ir_path, mode_dir / "candidate", expected_checksum)
    verify_path.write_text(json.dumps(verify_result, indent=2), encoding="utf-8")
    parsed = verify_result.get("parsed") or {}
    observed_time = parsed.get("time_sec")
    record.update(
        {
            "ok": verify_result["ok"],
            "compile_ok": verify_result["compile"]["ok"],
            "run_ok": bool(verify_result.get("run") and verify_result["run"]["ok"]),
            "checksum_ok": verify_result["checksum_ok"],
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
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--agent-cmd", default="", help="Compatibility path: command that reads prompt stdin and writes IR stdout.")
    parser.add_argument("--llm-config", "--config", dest="llm_config", type=Path, help="YAML/JSON config with optional llm: section.")
    parser.add_argument("--model", default="", help="Override config llm.llm_model / OPENAI_MODEL.")
    parser.add_argument("--api-mode", default="", choices=["", "auto", "responses", "chat"])
    parser.add_argument("--api-base-url", default="", help="Override config llm.base_url / OPENAI_BASE_URL.")
    parser.add_argument("--api-key", default="", help="Override config llm.api_key / OPENAI_API_KEY.")
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--truncation", default="auto")
    parser.add_argument("--store", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--base-backoff", type=float, default=1.0)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
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
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--scalar-opt-level", default="3", choices=["0", "1", "2", "3", "s", "z"])
    parser.add_argument("--extra-clang-arg", action="append", default=[])
    parser.add_argument("--skip-existing-ir", action="store_true")
    parser.add_argument("--max-recommendation-regions", type=int, default=8)
    parser.add_argument("--recommendation-top-k", type=int, default=20)
    parser.add_argument("--checksum-atol", type=float, default=1e-3)
    parser.add_argument("--build-timeout", type=int, default=60)
    parser.add_argument("--run-timeout", type=int, default=60)
    parser.add_argument("--agent-timeout", type=int, default=900)
    parser.add_argument("--prompt-only", action="store_true")
    args = parser.parse_args()

    if args.limit < 0:
        args.limit = 0
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

    modes = parse_modes(args.modes)
    template = args.template.read_text(encoding="utf-8")
    recommendations = load_recommendations(args.recommendations_jsonl)
    selected = select_tasks(load_jsonl(args.manifest), args.kernels, args.offset, args.limit)
    if not selected:
        parser.error("selected task range is empty")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "target.json").write_text(json.dumps(target_info, indent=2), encoding="utf-8")
    records_path = args.out_dir / "records.jsonl"
    records_path.write_text("", encoding="utf-8")
    records: list[dict[str, Any]] = []

    for task_index, task in selected:
        kernel = str(task["kernel"])
        task_dir = args.out_dir / f"{task_index:04d}_{kernel}"
        task_dir.mkdir(parents=True, exist_ok=True)
        scalar_ir_path = args.ir_dir / f"{kernel}.ll"

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
            with records_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps({"task_index": task_index, "kernel": kernel, "ok": False, "error": record["error"]}), flush=True)
            continue

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
            with records_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps({"task_index": task_index, "kernel": kernel, "ok": False, "error": record["error"]}), flush=True)
            continue

        expected_checksum = float(reference["parsed"]["checksum"])
        reference_time = float(reference["parsed"]["time_sec"])
        scalar_eval = build_and_run_ir(args, task, scalar_ir_path, task_dir / "scalar_ir", expected_checksum)
        (task_dir / "scalar_ir.json").write_text(json.dumps(scalar_eval, indent=2), encoding="utf-8")
        scalar_parsed = scalar_eval.get("parsed") or {}
        scalar_time = float(scalar_parsed["time_sec"]) if scalar_eval["ok"] and scalar_parsed.get("time_sec") else None
        scalar_ir = scalar_ir_path.read_text(encoding="utf-8", errors="replace")
        recommendation_rows = recommendations.get(kernel, [])

        record = {
            "task_index": task_index,
            "kernel": kernel,
            "scalar_ir": str(scalar_ir_path),
            "reference_time_sec": reference_time,
            "scalar_ir_time_sec": scalar_time,
            "expected_checksum": expected_checksum,
            "recommendation_regions": len(recommendation_rows),
            "modes": {},
            "ok": False,
        }
        for mode in modes:
            mode_record = run_mode(
                args,
                mode=mode,
                task=task,
                task_dir=task_dir,
                template=template,
                scalar_ir=scalar_ir,
                recommendation_rows=recommendation_rows,
                expected_checksum=expected_checksum,
                reference_time=reference_time,
                scalar_time=scalar_time,
            )
            record["modes"][mode] = mode_record
            print(
                json.dumps(
                    {
                        "task_index": task_index,
                        "kernel": kernel,
                        "mode": mode,
                        "ok": mode_record.get("ok", False),
                        "time_sec": mode_record.get("observed_time_sec"),
                        "speedup_vs_scalar_ir": mode_record.get("speedup_vs_scalar_ir"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        record["ok"] = any(mode_record.get("ok") for mode_record in record["modes"].values())
        records.append(record)
        with records_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")

    summary: dict[str, Any] = {
        "tasks": len(records),
        "modes": modes,
        "target": args.target,
        "target_cpu": args.target_cpu,
        "target_cpu_name": args.target_cpu_name,
        "target_info": str(args.out_dir / "target.json"),
        "prompt_only": args.prompt_only,
        "records": str(records_path),
        "successes": {
            mode: sum(1 for record in records if record.get("modes", {}).get(mode, {}).get("ok"))
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
            if baseline.get("ok") and guided.get("ok"):
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
