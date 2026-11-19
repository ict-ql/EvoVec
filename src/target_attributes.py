from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import Any


def _run(cmd: list[str], *, input_text: str = "", timeout: int = 30) -> dict[str, Any]:
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
    except Exception as exc:  # noqa: BLE001 - detection is best-effort.
        return {"ok": False, "cmd": cmd, "stdout": "", "stderr": str(exc), "error": str(exc)}
    return {
        "ok": proc.returncode == 0,
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def detect_clang_target_attributes(
    *,
    clang: str,
    target: str,
    march: str = "native",
    timeout: int = 30,
) -> dict[str, Any]:
    cmd = [clang]
    if target:
        cmd.extend(["--target", target])
    cmd.extend(["-x", "c", "-", "-S", "-emit-llvm", "-o", "-", f"-march={march}"])
    result = _run(cmd, input_text="void f(void) {}\n", timeout=timeout)
    text = str(result.get("stdout", ""))
    cpu = ""
    features = ""
    attrs_line = ""
    for line in text.splitlines():
        if line.startswith("attributes #") and "target-cpu" in line:
            attrs_line = line
        if "target-cpu" in line:
            match = re.search(r'"target-cpu"="([^"]+)"', line)
            if match:
                cpu = match.group(1)
        if "target-features" in line:
            match = re.search(r'"target-features"="([^"]+)"', line)
            if match:
                features = match.group(1)
    return {
        "ok": bool(result.get("ok") and (cpu or features)),
        "target_cpu": cpu,
        "target_features": features,
        "attribute_line": attrs_line,
        "detect_result": result,
    }


def features_to_set(features: str) -> set[str]:
    return {item[1:] for item in features.split(",") if item.startswith("+")}


def target_attribute_values(
    *,
    clang: str,
    target: str,
    target_cpu: str,
    fallback_features: list[str] | None = None,
    march: str = "native",
    timeout: int = 30,
) -> dict[str, Any]:
    detected = detect_clang_target_attributes(
        clang=clang,
        target=target,
        march=march,
        timeout=timeout,
    )
    cpu = str(detected.get("target_cpu") or "").strip()
    if cpu in {"", "native", "generic"} and target_cpu:
        cpu = target_cpu
    features = str(detected.get("target_features") or "").strip()
    if not features and fallback_features:
        features = ",".join(fallback_features)
    attrs: dict[str, str] = {}
    if cpu:
        attrs["target-cpu"] = cpu
    if features:
        attrs["target-features"] = features
    return {
        "attributes": attrs,
        "target_features_set": sorted(features_to_set(features)),
        "detection": detected,
    }


ATTR_GROUP_RE = re.compile(r"^(?P<prefix>attributes\s+#(?P<id>\d+)\s*=\s*\{)(?P<body>.*)(?P<suffix>\}\s*)$")
DEFINE_ATTR_RE = re.compile(r"^(?P<prefix>\s*define\b.*?)(?:\s+#(?P<id>\d+))?(?P<suffix>\s*\{.*)$")


def _escape_attr_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _replace_or_append_attr(body: str, key: str, value: str) -> str:
    escaped = _escape_attr_value(value)
    pattern = re.compile(rf'\s*"{re.escape(key)}"="[^"]*"')
    replacement = f' "{key}"="{escaped}"'
    if pattern.search(body):
        return pattern.sub(replacement, body, count=1).strip()
    return (body.strip() + replacement).strip()


def _patch_attr_body(body: str, attrs: dict[str, str]) -> str:
    for key, value in attrs.items():
        body = _replace_or_append_attr(body, key, value)
    return body


def _next_attr_id(lines: list[str]) -> int:
    max_id = -1
    for line in lines:
        match = ATTR_GROUP_RE.match(line.strip())
        if match:
            max_id = max(max_id, int(match.group("id")))
    return max_id + 1


def patch_ir_target_attributes(ir_text: str, attrs: dict[str, str]) -> tuple[str, dict[str, Any]]:
    if not attrs:
        return ir_text, {"patched": False, "reason": "empty target attributes"}

    lines = ir_text.splitlines()
    define_attr_ids: set[int] = set()
    define_lines_without_attrs: list[int] = []
    for index, line in enumerate(lines):
        match = DEFINE_ATTR_RE.match(line)
        if not match:
            continue
        attr_id = match.group("id")
        if attr_id is None:
            define_lines_without_attrs.append(index)
        else:
            define_attr_ids.add(int(attr_id))

    if not define_attr_ids and not define_lines_without_attrs:
        return ir_text, {"patched": False, "reason": "no function definitions"}

    created_attr_id: int | None = None
    if define_lines_without_attrs:
        created_attr_id = _next_attr_id(lines)
        for index in define_lines_without_attrs:
            match = DEFINE_ATTR_RE.match(lines[index])
            if match:
                lines[index] = f"{match.group('prefix')} #{created_attr_id}{match.group('suffix')}"
        define_attr_ids.add(created_attr_id)

    patched_group_ids: set[int] = set()
    existing_group_ids: set[int] = set()
    for index, line in enumerate(lines):
        match = ATTR_GROUP_RE.match(line.strip())
        if not match:
            continue
        group_id = int(match.group("id"))
        existing_group_ids.add(group_id)
        if group_id not in define_attr_ids:
            continue
        body = _patch_attr_body(match.group("body"), attrs)
        lines[index] = f'{match.group("prefix")} {body} {match.group("suffix")}'.strip()
        patched_group_ids.add(group_id)

    missing_group_ids = sorted(define_attr_ids.difference(existing_group_ids))
    for group_id in missing_group_ids:
        body = _patch_attr_body("", attrs)
        lines.append(f"attributes #{group_id} = {{ {body} }}")
        patched_group_ids.add(group_id)

    newline = "\n" if ir_text.endswith("\n") else ""
    return "\n".join(lines) + newline, {
        "patched": True,
        "attrs": attrs,
        "patched_attribute_groups": sorted(patched_group_ids),
        "created_attribute_group": created_attr_id,
        "definitions_without_attrs": len(define_lines_without_attrs),
    }


def patch_ir_file(path: Path, attrs: dict[str, str]) -> dict[str, Any]:
    original = path.read_text(encoding="utf-8", errors="replace")
    patched, summary = patch_ir_target_attributes(original, attrs)
    if patched != original:
        path.write_text(patched, encoding="utf-8")
    summary["path"] = str(path)
    summary["changed"] = patched != original
    return summary
