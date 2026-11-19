from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"missing LLM config: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyYAML is required for YAML config files; install pyyaml or pass JSON.") from exc
        value = yaml.safe_load(text) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping config: {path}")
    return value


def llm_section(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("llm", config)
    return section if isinstance(section, dict) else {}


def config_value(section: dict[str, Any], key: str, env_key: str, default: Any = "") -> Any:
    value = section.get(key)
    if value not in (None, ""):
        return value
    return os.environ.get(env_key, default)


class LLMClient:
    def __init__(self, *, base_url: str, api_key: str):
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("The openai Python package is required for LLM calls.") from exc
        if not api_key:
            raise ValueError("missing API key; set config llm.api_key or OPENAI_API_KEY")
        self.client = OpenAI(base_url=base_url or None, api_key=api_key)

    @staticmethod
    def _is_not_found_404(exc: Exception) -> bool:
        text = str(exc).lower()
        type_name = type(exc).__name__.lower()
        return "notfound" in type_name or ("404" in text and "not found" in text)

    @staticmethod
    def _extract_chat_text(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            return str(response)
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "")
        return content or ""

    def _via_responses(
        self,
        *,
        prompt_text: str,
        model: str,
        max_output_tokens: int,
        temperature: float | None,
        truncation: str,
        store: bool,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "input": prompt_text,
            "max_output_tokens": max_output_tokens,
            "truncation": truncation,
            "store": store,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self.client.responses.create(**kwargs)
        text = getattr(response, "output_text", None)
        result = str(response) if text is None else str(text)
        if not result.strip():
            raise RuntimeError("responses API returned empty output")
        return result

    def _via_chat(
        self,
        *,
        prompt_text: str,
        model: str,
        temperature: float | None,
        store: bool,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
            "store": store,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self.client.chat.completions.create(**kwargs)
        result = self._extract_chat_text(response)
        if not result.strip():
            raise RuntimeError("chat completions API returned empty output")
        return result

    def call(
        self,
        *,
        prompt_text: str,
        model: str,
        max_output_tokens: int,
        temperature: float | None = None,
        truncation: str = "auto",
        store: bool = False,
        api_mode: str = "auto",
        max_retries: int = 2,
        base_backoff: float = 1.0,
    ) -> str:
        for attempt in range(max_retries + 1):
            try:
                if api_mode == "responses":
                    return self._via_responses(
                        prompt_text=prompt_text,
                        model=model,
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                        truncation=truncation,
                        store=store,
                    )
                if api_mode == "chat":
                    return self._via_chat(
                        prompt_text=prompt_text,
                        model=model,
                        temperature=temperature,
                        store=store,
                    )
                try:
                    return self._via_responses(
                        prompt_text=prompt_text,
                        model=model,
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                        truncation=truncation,
                        store=store,
                    )
                except Exception as exc:  # noqa: BLE001 - fallback mode mirrors ask_llm.
                    if self._is_not_found_404(exc) or "empty output" in str(exc).lower():
                        return self._via_chat(
                            prompt_text=prompt_text,
                            model=model,
                            temperature=temperature,
                            store=store,
                        )
                    raise
            except Exception as exc:  # noqa: BLE001 - classify retryable API failures.
                text = str(exc).lower()
                retryable = any(
                    marker in text
                    for marker in (
                        "rate limit",
                        "429",
                        "timeout",
                        "timed out",
                        "temporarily",
                        "temporary",
                        "connection",
                        "reset",
                        "502",
                        "503",
                        "504",
                        "server error",
                        "internal error",
                    )
                )
                if not retryable or attempt >= max_retries:
                    raise
                time.sleep(base_backoff * (2**attempt) + random.random() * 0.25)
        raise RuntimeError("unreachable retry loop")


def client_from_config(
    config_path: Path | None,
    *,
    base_url: str = "",
    api_key: str = "",
) -> tuple[LLMClient, dict[str, Any]]:
    config = load_config(config_path)
    section = llm_section(config)
    resolved_base_url = base_url or str(config_value(section, "base_url", "OPENAI_BASE_URL", ""))
    resolved_api_key = api_key or str(config_value(section, "api_key", "OPENAI_API_KEY", ""))
    return LLMClient(base_url=resolved_base_url, api_key=resolved_api_key), section
