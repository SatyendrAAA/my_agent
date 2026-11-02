"""Configuration for my_agent.

Everything can be set from environment variables so the CLI stays zero-config:

    MY_AGENT_API_KEY / OPENAI_API_KEY     API key for an OpenAI-compatible endpoint
    MY_AGENT_BASE_URL / OPENAI_BASE_URL   default: https://api.openai.com/v1
    MY_AGENT_MODEL / OPENAI_MODEL         default: gpt-4o-mini
    MY_AGENT_MAX_STEPS                    tool-call rounds per turn (default 8)
    MY_AGENT_ALLOW_SHELL                  1/true to enable the run_shell tool
    MY_AGENT_SESSION_DIR                  where sessions are stored
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SESSION_DIR = Path.home() / ".my_agent" / "sessions"

SYSTEM_PROMPT = """You are my_agent, a compact command-line assistant running in the user's terminal.

You can use tools to inspect and change files, run commands, and execute Python.
Rules:
- Prefer using tools over guessing. Read a file before editing it.
- Keep answers short and concrete; show the command or the diff you used.
- All file paths are relative to the working directory {workdir}.
- If a tool returns an error, adapt instead of repeating the same call.
- Never invent file contents you have not read.
"""

_TRUE = {"1", "true", "yes", "y", "on"}


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return default


def _flag(*names: str) -> bool:
    return _env(*names).lower() in _TRUE


@dataclass
class Config:
    """Runtime settings shared by the CLI, the agent loop and the tools."""

    workdir: Path = field(default_factory=lambda: Path.cwd().resolve())
    model: str = field(default_factory=lambda: _env("MY_AGENT_MODEL", "OPENAI_MODEL", default=DEFAULT_MODEL))
    base_url: str = field(
        default_factory=lambda: _env("MY_AGENT_BASE_URL", "OPENAI_BASE_URL", default=DEFAULT_BASE_URL)
    )
    api_key: str = field(default_factory=lambda: _env("MY_AGENT_API_KEY", "OPENAI_API_KEY"))
    session_dir: Path = field(
        default_factory=lambda: Path(_env("MY_AGENT_SESSION_DIR", default=str(DEFAULT_SESSION_DIR))).expanduser()
    )
    max_steps: int = field(default_factory=lambda: int(_env("MY_AGENT_MAX_STEPS", default="8") or 8))
    allow_shell: bool = field(default_factory=lambda: _flag("MY_AGENT_ALLOW_SHELL"))
    offline: bool = False
    temperature: float = 0.2
    timeout: float = 120.0
    max_output_chars: int = 8000

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir).expanduser().resolve()
        self.session_dir = Path(self.session_dir).expanduser()
        self.max_steps = max(1, int(self.max_steps))
        self.base_url = self.base_url.rstrip("/")

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(workdir=self.workdir)

    def tool_names_allowed(self) -> set[str]:
        names = {"list_dir", "read_file", "write_file", "search_files", "python_eval"}
        if self.allow_shell:
            names.add("run_shell")
        return names
