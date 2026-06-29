#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neural_asm_pipeline.llm_client import client_from_config, config_value  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick OpenAI-compatible LLM query.")
    parser.add_argument("--config", type=Path, help="YAML/JSON config with an optional llm: section.")
    parser.add_argument("-p", "--prompt", default="", help="Inline prompt text.")
    parser.add_argument("--file", type=Path, help="Read prompt from a file.")
    parser.add_argument("--model", default="", help="Override model name.")
    parser.add_argument("--api-mode", default="", choices=["", "auto", "responses", "chat"])
    parser.add_argument("--api-base-url", default="", help="Override config llm.base_url / OPENAI_BASE_URL.")
    parser.add_argument("--api-key", default="", help="Override config llm.api_key / OPENAI_API_KEY.")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--truncation", default="auto")
    parser.add_argument("--store", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--base-backoff", type=float, default=1.0)
    args = parser.parse_args()

    if args.file:
        prompt = args.file.read_text(encoding="utf-8")
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = sys.stdin.read()
    if not prompt.strip():
        raise SystemExit("empty prompt")

    client, section = client_from_config(args.config, base_url=args.api_base_url, api_key=args.api_key)
    model = args.model or str(config_value(section, "llm_model", "OPENAI_MODEL", "gpt-5"))
    api_mode = args.api_mode or str(config_value(section, "api_mode", "OPENAI_API_MODE", "auto"))
    print(f"[Model: {model}  Tokens: {args.max_tokens}  API: {api_mode}]", file=sys.stderr)
    response = client.call(
        prompt_text=prompt,
        model=model,
        max_output_tokens=args.max_tokens,
        temperature=args.temperature,
        truncation=args.truncation,
        store=args.store,
        api_mode=api_mode,
        max_retries=args.max_retries,
        base_backoff=args.base_backoff,
    )
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
