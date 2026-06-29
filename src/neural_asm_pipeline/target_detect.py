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


def detect_target(
    *,
    compiler: str,
    target: str,
    target_cpu: str = "auto",
    target_cpu_name: str = "",
    timeout: int = 10,
) -> dict[str, Any]:
    detected = clang_native_target(compiler, target, timeout) if target_cpu == "auto" else {}
    llvm_cpu = str(detected.get("target_cpu") or "").strip()
    if target_cpu and target_cpu != "auto":
        llvm_cpu = target_cpu
    if not llvm_cpu:
        llvm_cpu = "native"
    model_name = target_cpu_name.strip() if target_cpu_name else cpu_model_name(timeout)
    return {
        "target": target,
        "target_cpu": llvm_cpu,
        "target_cpu_name": model_name,
        "target_features": detected.get("target_features", []),
        "detection": detected,
    }
