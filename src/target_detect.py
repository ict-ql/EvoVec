from __future__ import annotations

import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any


def _run(cmd: list[str], timeout: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - detection must be best-effort.
        return {"ok": False, "cmd": cmd, "error": str(exc), "stdout": "", "stderr": ""}
    return {
        "ok": proc.returncode == 0,
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def cpu_model_name(timeout: int = 5) -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware")) and ":" in line:
                value = line.split(":", 1)[1].strip()
                if value:
                    return value

    if platform.system() == "Darwin":
        result = _run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout)
        if result["ok"] and str(result["stdout"]).strip():
            return str(result["stdout"]).strip()

    env_value = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
    if env_value:
        return env_value
    return platform.processor() or platform.machine() or "unknown"


def parse_clang_target_cpu(text: str) -> str:
    patterns = (
        r'["\']-target-cpu["\']\s+["\']([^"\']+)["\']',
        r"(?:^|\s)-target-cpu\s+([^\s\"']+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def parse_clang_target_features(text: str) -> list[str]:
    features: list[str] = []
    for match in re.finditer(r'["\']-target-feature["\']\s+["\']([^"\']+)["\']', text):
        features.append(match.group(1))
    return features


def is_generic_target_cpu(value: str) -> bool:
    return value.strip().lower() in {"", "native", "generic"}


def clang_native_target(compiler: str, target: str, timeout: int = 10) -> dict[str, Any]:
    cmd = [
        compiler,
        "-###",
        "-c",
        "-x",
        "c",
        os.devnull,
        "-march=native",
        "-o",
        os.devnull,
    ]
    if target:
        cmd[1:1] = ["--target", target]
    result = _run(cmd, timeout)
    text = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
    result["target_cpu"] = parse_clang_target_cpu(text)
    result["target_features"] = parse_clang_target_features(text)
    return result


def parse_gcc_march(text: str) -> str:
    for line in text.splitlines():
        match = re.match(r"\s*-march=\s*([A-Za-z0-9_.+-]+)", line)
        if match:
            return match.group(1)
    match = re.search(r"(?:^|\s)-march=([A-Za-z0-9_.+-]+)", text)
    return match.group(1) if match else ""


def gcc_native_arch(timeout: int = 10) -> dict[str, Any]:
    for compiler in ("gcc", "cc"):
        result = _run([compiler, "-Q", "--help=target", "-march=native"], timeout)
        text = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
        march = parse_gcc_march(text)
        result["target_cpu"] = march
        if result["ok"] and march and not is_generic_target_cpu(march):
            return result
    return {"ok": False, "target_cpu": "", "error": "no resolved GCC -march=native value"}


def intel_xeon_scalable_cpu(model: str) -> str:
    match = re.search(r"\b(?:platinum|gold|silver|bronze)\s+([0-9]{4,5})", model)
    if not match:
        return ""
    sku = match.group(1)
    family = int(sku[:2])
    if family in {84, 64, 54, 44}:
        return "sapphirerapids"
    if family in {83, 63, 53, 43, 33}:
        return "icelake-server"
    if family in {82, 62, 52, 42, 32}:
        return "cascadelake"
    if family in {81, 61, 51, 41, 31}:
        return "skylake-avx512"
    return ""


def cpu_model_to_llvm_cpu(model_name: str) -> str:
    model = model_name.lower()
    direct_patterns = (
        ("sapphire rapids", "sapphirerapids"),
        ("ice lake", "icelake-server" if "xeon" in model else "icelake-client"),
        ("cascade lake", "cascadelake"),
        ("skylake", "skylake-avx512" if "xeon" in model else "skylake"),
        ("broadwell", "broadwell"),
        ("haswell", "haswell"),
        ("ivy bridge", "ivybridge"),
        ("sandy bridge", "sandybridge"),
        ("alder lake", "alderlake"),
        ("rocket lake", "rocketlake"),
        ("tiger lake", "tigerlake"),
        ("coffee lake", "coffeelake"),
        ("kaby lake", "kabylake"),
        ("zen 5", "znver5"),
        ("zen 4", "znver4"),
        ("zen 3", "znver3"),
        ("zen 2", "znver2"),
    )
    for needle, cpu in direct_patterns:
        if needle in model:
            return cpu

    if "intel" in model and "xeon" in model:
        scalable = intel_xeon_scalable_cpu(model)
        if scalable:
            return scalable
        version_match = re.search(r"\bv([234])\b", model)
        if version_match:
            return {"2": "ivybridge", "3": "haswell", "4": "broadwell"}[version_match.group(1)]

    if "amd" in model or "epyc" in model or "ryzen" in model:
        epyc_match = re.search(r"\b(?:epyc|epyc\(tm\))\s+([0-9]{4})", model)
        if epyc_match:
            sku = epyc_match.group(1)
            if sku.startswith("9") and sku.endswith("5"):
                return "znver5"
            if sku.startswith("9") or sku.endswith("4"):
                return "znver4"
            if sku.endswith("3"):
                return "znver3"
            if sku.endswith("2"):
                return "znver2"
        ryzen_match = re.search(r"\bryzen(?:\([a-z]+\))?\s+\d\s+([0-9]{4})", model)
        if ryzen_match:
            generation = ryzen_match.group(1)[0]
            if generation in {"8", "7"}:
                return "znver4"
            if generation == "5":
                return "znver3"
            if generation == "3":
                return "znver2"
    return ""


def detect_target(
    *,
    compiler: str,
    target: str,
    target_cpu: str = "auto",
    target_cpu_name: str = "",
    timeout: int = 10,
) -> dict[str, Any]:
    detected = clang_native_target(compiler, target, timeout) if target_cpu == "auto" else {}
    model_name = target_cpu_name.strip() if target_cpu_name else cpu_model_name(timeout)
    llvm_cpu = str(detected.get("target_cpu") or "").strip()
    source = "clang-march-native"
    gcc_detected: dict[str, Any] = {}
    if target_cpu and target_cpu != "auto":
        llvm_cpu = target_cpu
        source = "user"
    elif is_generic_target_cpu(llvm_cpu):
        gcc_detected = gcc_native_arch(timeout)
        gcc_cpu = str(gcc_detected.get("target_cpu") or "").strip()
        if not is_generic_target_cpu(gcc_cpu):
            llvm_cpu = gcc_cpu
            source = "gcc-march-native"
        else:
            model_cpu = cpu_model_to_llvm_cpu(model_name)
            if model_cpu:
                llvm_cpu = model_cpu
                source = "cpu-model-heuristic"
            else:
                llvm_cpu = "x86-64-v3" if target.startswith("x86_64") else "native"
                source = "generic-fallback"
    return {
        "target": target,
        "target_cpu": llvm_cpu,
        "target_cpu_source": source,
        "target_cpu_name": model_name,
        "target_features": detected.get("target_features", []),
        "detection": detected,
        "gcc_detection": gcc_detected,
    }
