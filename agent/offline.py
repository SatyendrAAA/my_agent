"""A tiny deterministic "model" so the agent is usable with no API key.

This is NOT an LLM. It pattern-matches a handful of English/Hinglish commands
onto the same tool-call format a real model uses, which makes the whole loop
(`plan -> tool -> observe -> answer`) runnable and testable offline.
"""

from __future__ import annotations

import json
import re
import shlex
from typing import Any

HELP = """Offline mode: no API key configured, so I only understand a few command shapes.

  ls / list                list the working directory
  read <file>              print a file with line numbers
  search <text>            grep the working directory
  python: <code>           run a Python snippet
  run <command>            run a shell command (needs --allow-shell)
  write <file> :: <text>   create/overwrite a file

Anything else needs a real model:
  export OPENAI_API_KEY=sk-...      (or MY_AGENT_API_KEY, MY_AGENT_BASE_URL, MY_AGENT_MODEL)

Type /help for the agent's own commands.
"""

_LIST_RE = re.compile(r"^(ls|dir|list|files|tree)\b", re.IGNORECASE)
_READ_RE = re.compile(
    r"^(?:read|cat|show|open|padho|dikhao|kholo|parho)\s+(?P<path>\S.*)$", re.IGNORECASE
)
_SEARCH_RE = re.compile(r"^(?:search|grep|find|dhundo|khojo)\s+(?P<query>.+)$", re.IGNORECASE)
_PYTHON_RE = re.compile(r"^(?:python|py|run python)\s*:\s*(?P<code>.+)$", re.IGNORECASE | re.DOTALL)
_SHELL_RE = re.compile(r"^(?:run|sh|shell|exec|chalao)\s+(?P<command>.+)$", re.IGNORECASE | re.DOTALL)
_WRITE_RE = re.compile(r"^write\s+(?P<path>\S+)\s*::\s*(?P<content>.*)$", re.IGNORECASE | re.DOTALL)


class OfflineModel:
    """Deterministic stand-in that implements the same `chat` contract."""

    name = "offline-planner"

    def __init__(self) -> None:
        self._calls = 0

    # ------------------------------------------------------------------ contract
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        if messages and messages[-1].get("role") == "tool":
            return {"role": "assistant", "content": self._summarise(messages), "tool_calls": []}

        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"),
            None,
        )
        text = (last_user or {}).get("content") or ""
        call = self._plan(text.strip())
        if call is None:
            return {"role": "assistant", "content": HELP, "tool_calls": []}
        return {"role": "assistant", "content": "", "tool_calls": [call]}

    # ------------------------------------------------------------------- planning
    def _plan(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None

        if _LIST_RE.match(text):
            return self._call("list_dir", {"path": "."})

        match = _READ_RE.match(text)
        if match:
            return self._call("read_file", {"path": _clean_path(match.group("path"))})

        match = _SEARCH_RE.match(text)
        if match:
            return self._call("search_files", {"query": match.group("query").strip()})

        match = _PYTHON_RE.match(text)
        if match:
            return self._call("python_eval", {"code": match.group("code").strip()})

        match = _WRITE_RE.match(text)
        if match:
            return self._call(
                "write_file",
                {"path": match.group("path"), "content": match.group("content")},
            )

        match = _SHELL_RE.match(text)
        if match:
            return self._call("run_shell", {"command": match.group("command").strip()})

        return None

    def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._calls += 1
        return {
            "id": f"offline_call_{self._calls}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }

    # ------------------------------------------------------------------ observing
    @staticmethod
    def _summarise(messages: list[dict[str, Any]]) -> str:
        last_user_index = max(
            (i for i, m in enumerate(messages) if m.get("role") == "user"),
            default=-1,
        )
        observations = [m for m in messages[last_user_index + 1 :] if m.get("role") == "tool"]
        if not observations:
            return "(tool finished with no output)"
        chunks = []
        for message in observations:
            body = (message.get("content") or "").rstrip() or "(no output)"
            chunks.append(body)
        return "\n\n".join(chunks)


def _clean_path(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    try:
        parts = shlex.split(raw)
    except ValueError:
        return raw
    return parts[0] if parts else raw
