"""Minimal OpenAI-compatible chat client (stdlib only, no SDK required).

Works with OpenAI, Azure-style gateways, Ollama (`/v1`), LM Studio, vLLM,
Together, Groq, ... anything that speaks `POST /chat/completions`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol


class LLMError(RuntimeError):
    """Raised when the model endpoint cannot be reached or returns garbage."""


class ChatModel(Protocol):
    """Anything that can answer one chat-completion request."""

    name: str

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Return an assistant message dict: {"role": "assistant", "content": str, "tool_calls": [...]}."""


class OpenAICompatibleModel:
    """Talks to any `/chat/completions` endpoint over plain HTTP."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.2,
        timeout: float = 120.0,
    ) -> None:
        self.name = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    # ------------------------------------------------------------------ public
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        data = self._post("/chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"model returned no choices: {json.dumps(data)[:500]}")
        message = choices[0].get("message") or {}
        return {
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": message.get("tool_calls") or [],
        }

    def list_models(self) -> list[str]:
        try:
            data = self._post("/models", None, method="GET")
        except LLMError:
            return []
        return sorted(item.get("id", "") for item in data.get("data", []) if item.get("id"))

    # ----------------------------------------------------------------- private
    def _post(self, path: str, payload: dict[str, Any] | None, method: str = "POST") -> dict[str, Any]:
        body_bytes = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.base_url + path, data=body_bytes, method=method)
        request.add_header("Content-Type", "application/json")
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        request.add_header("User-Agent", "my-agent/0.1")

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:800]
            raise LLMError(f"HTTP {exc.code} from {self.base_url}{path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"cannot reach {self.base_url}{path}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMError(f"request to {self.base_url}{path} timed out after {self.timeout}s") from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"invalid JSON from {self.base_url}{path}: {raw[:300]}") from exc
