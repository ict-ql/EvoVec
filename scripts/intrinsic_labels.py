from __future__ import annotations

from collections import Counter
import re


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
