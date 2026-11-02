"""The agent loop: ask the model, run its tools, feed results back, repeat."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Config
from .llm import ChatModel
from .memory import Session
from .tools import ToolRegistry

EventHook = Callable[[str, dict[str, Any]], None]


@dataclass
class TurnResult:
    """What one user turn produced."""

    content: str
    steps: int = 0
    tool_names: list[str] = field(default_factory=list)
    truncated: bool = False


class Agent:
    """Wires a model, a tool registry and a session together."""

    def __init__(
        self,
        config: Config,
        model: ChatModel,
        tools: ToolRegistry,
        session: Session,
        on_event: EventHook | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.tools = tools
        self.session = session
        self.on_event = on_event

    # ------------------------------------------------------------------ public
    def handle(self, user_text: str) -> TurnResult:
        """Run one full turn for `user_text` and return the final answer."""
        user_text = user_text.strip()
        if not user_text:
            return TurnResult(content="(empty message)")

        self.session.append({"role": "user", "content": user_text})
        self._emit("turn_start", {"text": user_text})

        used: list[str] = []
        last_text = ""

        for step in range(1, self.config.max_steps + 1):
            self._emit("thinking", {"step": step})
            message = self.model.chat(self.session.history(), self.tools.schemas())
            content = (message.get("content") or "").strip()
            calls = message.get("tool_calls") or []
            if content:
                last_text = content

            if not calls:
                self.session.append({"role": "assistant", "content": content})
                self._emit("assistant", {"text": content})
                return TurnResult(content=content, steps=step, tool_names=used)

            self.session.append(
                {"role": "assistant", "content": content, "tool_calls": calls}
            )

            for index, call in enumerate(calls):
                name, arguments, call_id = _parse_call(call, index)
                used.append(name)
                self._emit("tool_call", {"name": name, "arguments": arguments})
                result = self.tools.call(name, arguments)
                self._emit("tool_result", {"name": name, "result": result})
                self.session.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": result,
                    }
                )

        # Ran out of steps: report the best text we have instead of hanging.
        note = (
            f"(stopped after {self.config.max_steps} tool rounds - raise --max-steps if the task needs more)\n\n"
            f"{last_text}".strip()
        )
        self.session.append({"role": "assistant", "content": note})
        self._emit("assistant", {"text": note})
        return TurnResult(content=note, steps=self.config.max_steps, tool_names=used, truncated=True)

    def reset(self) -> None:
        self.session.clear()
        self.session.system_prompt = self.config.system_prompt()

    # ----------------------------------------------------------------- private
    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(kind, payload)


def _parse_call(call: dict[str, Any], index: int) -> tuple[str, dict[str, Any] | str, str]:
    """Normalise the several shapes a tool call can arrive in."""
    if not isinstance(call, dict):
        return "", "", f"call_{index}"

    call_id = str(call.get("id") or f"call_{index}")
    function = call.get("function") or {}
    name = function.get("name") or call.get("name") or ""
    arguments: dict[str, Any] | str = function.get("arguments")
    if arguments is None:
        arguments = call.get("arguments") or {}

    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            arguments = {}
        else:
            try:
                parsed = json.loads(stripped)
                arguments = parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                arguments = stripped  # registry will report a readable error

    return name, arguments, call_id
