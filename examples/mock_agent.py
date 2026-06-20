#!/usr/bin/env python3
from __future__ import annotations

import sys


def main() -> int:
    _prompt = sys.stdin.read()
    print(
        """# Mock agent output.
# Replace examples/mock_agent.py with a real LLM compiler agent.
.text
.globl llm_compiler_placeholder
llm_compiler_placeholder:
    ret
"""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
