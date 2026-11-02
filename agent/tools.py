"""Tool registry plus the built-in, sandboxed tools the agent can call.

Every tool is a plain Python function that takes a dict of arguments and returns
a string. Paths are always resolved inside `config.workdir`; anything that
escapes it (``../`` or absolute paths outside the tree) is rejected.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import Config

MAX_READ_LINES = 400
MAX_READ_BYTES = 200_000
DEFAULT_TIMEOUT = 30


class ToolError(RuntimeError):
    """A tool failed in a way the model should see (and can recover from)."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[[dict[str, Any], "ToolContext"], str]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolContext:
    config: Config
    confirm: Callable[[str], bool] | None = None


# --------------------------------------------------------------------- helpers
def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} chars omitted]"


def _resolve(ctx: ToolContext, raw_path: str | None, *, must_exist: bool = False) -> Path:
    root = ctx.config.workdir
    target = Path(raw_path or ".").expanduser()
    path = (target if target.is_absolute() else root / target).resolve()
    if path != root and root not in path.parents:
        raise ToolError(f"path escapes the working directory: {raw_path} (workdir={root})")
    if must_exist and not path.exists():
        raise ToolError(f"no such file or directory: {raw_path}")
    return path


def _is_probably_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(4096)
    except OSError:
        return False
    return b"\x00" in chunk


# ------------------------------------------------------------------------ tools
def list_dir(args: dict[str, Any], ctx: ToolContext) -> str:
    path = _resolve(ctx, args.get("path"), must_exist=True)
    if not path.is_dir():
        raise ToolError(f"not a directory: {args.get('path')}")
    entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    if not entries:
        return f"{path} is empty"
    lines = []
    for entry in entries[:300]:
        try:
            size = entry.stat().st_size
        except OSError:
            size = 0
        kind = "dir " if entry.is_dir() else "file"
        lines.append(f"{kind}  {size:>9}  {entry.name}")
    if len(entries) > 300:
        lines.append(f"... {len(entries) - 300} more entries")
    return "\n".join(lines)


def read_file(args: dict[str, Any], ctx: ToolContext) -> str:
    path = _resolve(ctx, args.get("path"), must_exist=True)
    if path.is_dir():
        raise ToolError(f"{args.get('path')} is a directory; use list_dir")
    if path.stat().st_size > MAX_READ_BYTES:
        raise ToolError(f"{args.get('path')} is larger than {MAX_READ_BYTES} bytes")
    if _is_probably_binary(path):
        raise ToolError(f"{args.get('path')} looks binary")

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ToolError(f"cannot read {args.get('path')}: {exc}") from exc

    lines = text.splitlines()
    start = max(1, int(args.get("start_line") or 1))
    count = min(int(args.get("max_lines") or MAX_READ_LINES), MAX_READ_LINES)
    window = lines[start - 1 : start - 1 + count]
    numbered = "\n".join(f"{start + i:>5}  {line}" for i, line in enumerate(window))
    footer = ""
    if start - 1 + count < len(lines):
        footer = f"\n... ({len(lines) - (start - 1 + count)} more lines)"
    return _clip(numbered + footer or "(empty file)", ctx.config.max_output_chars)


def write_file(args: dict[str, Any], ctx: ToolContext) -> str:
    path = _resolve(ctx, args.get("path"))
    content = args.get("content")
    if not isinstance(content, str):
        raise ToolError("'content' must be a string")
    if path.exists() and path.is_dir():
        raise ToolError(f"{args.get('path')} is a directory")

    created = not path.exists()
    previous_lines = path.read_text(encoding="utf-8", errors="replace").count("\n") + 1 if path.exists() else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"cannot write {args.get('path')}: {exc}") from exc

    new_lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
    verb = "created" if created else "updated"
    return f"{verb} {args.get('path')} ({new_lines} lines, {len(content)} bytes; was {previous_lines} lines)"


def search_files(args: dict[str, Any], ctx: ToolContext) -> str:
    needle = args.get("query")
    if not needle:
        raise ToolError("'query' is required")
    root = _resolve(ctx, args.get("path"), must_exist=True)
    use_regex = bool(args.get("regex"))
    limit = min(int(args.get("max_results") or 60), 200)

    import re

    try:
        pattern = re.compile(needle if use_regex else re.escape(needle), re.IGNORECASE)
    except re.error as exc:
        raise ToolError(f"bad regex: {exc}") from exc

    skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build", ".mypy_cache"}
    files = [root] if root.is_file() else sorted(root.rglob("*"))
    hits: list[str] = []
    for file in files:
        if len(hits) >= limit:
            break
        if not file.is_file() or any(part in skip_dirs for part in file.parts):
            continue
        if file.stat().st_size > MAX_READ_BYTES or _is_probably_binary(file):
            continue
        try:
            for number, line in enumerate(file.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pattern.search(line):
                    rel = file.relative_to(ctx.config.workdir)
                    hits.append(f"{rel}:{number}: {line.strip()[:200]}")
                    if len(hits) >= limit:
                        break
        except OSError:
            continue
    if not hits:
        return f"no matches for {needle!r} in {root}"
    return "\n".join(hits)


def python_eval(args: dict[str, Any], ctx: ToolContext) -> str:
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ToolError("'code' is required")
    timeout = min(int(args.get("timeout") or DEFAULT_TIMEOUT), 120)
    return _run([sys.executable, "-I", "-c", code], ctx, timeout, label="python")


def run_shell(args: dict[str, Any], ctx: ToolContext) -> str:
    if not ctx.config.allow_shell:
        raise ToolError("shell access is disabled; restart with --allow-shell to enable it")
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ToolError("'command' is required")
    if ctx.confirm is not None and not ctx.confirm(command):
        raise ToolError("user rejected this command")

    timeout = min(int(args.get("timeout") or DEFAULT_TIMEOUT), 300)
    return _run(["/bin/sh", "-c", command], ctx, timeout, label="shell")


def _run(argv: list[str], ctx: ToolContext, timeout: int, *, label: str) -> str:
    try:
        completed = subprocess.run(
            argv,
            cwd=str(ctx.config.workdir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"{label} command timed out after {timeout}s") from None
    except OSError as exc:
        raise ToolError(f"{label} command could not start: {exc}") from exc

    parts = []
    if completed.stdout.strip():
        parts.append(completed.stdout.rstrip())
    if completed.stderr.strip():
        parts.append("[stderr]\n" + completed.stderr.rstrip())
    if completed.returncode != 0:
        parts.append(f"[exit code {completed.returncode}]")
    result = "\n".join(parts) or "(no output)"
    return _clip(result, ctx.config.max_output_chars)


# ---------------------------------------------------------------------- registry
ALL_TOOLS: tuple[Tool, ...] = (
    Tool(
        name="list_dir",
        description="List files and directories at a path inside the working directory.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path, default '.'"}},
        },
        func=list_dir,
    ),
    Tool(
        name="read_file",
        description="Read a UTF-8 text file with line numbers. Prefer this before editing anything.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the working directory"},
                "start_line": {"type": "integer", "description": "1-based first line (default 1)"},
                "max_lines": {"type": "integer", "description": "How many lines to return (default 400)"},
            },
            "required": ["path"],
        },
        func=read_file,
    ),
    Tool(
        name="write_file",
        description="Create or overwrite a text file with the given content. Creates parent directories.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the working directory"},
                "content": {"type": "string", "description": "Full new file content"},
            },
            "required": ["path", "content"],
        },
        func=write_file,
    ),
    Tool(
        name="search_files",
        description="Search file contents under a path (case-insensitive substring or regex).",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text or regex to look for"},
                "path": {"type": "string", "description": "File or directory to search, default '.'"},
                "regex": {"type": "boolean", "description": "Treat query as a regular expression"},
                "max_results": {"type": "integer", "description": "Stop after N hits (default 60)"},
            },
            "required": ["query"],
        },
        func=search_files,
    ),
    Tool(
        name="python_eval",
        description="Run a short Python snippet in an isolated interpreter and return its output.",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute"},
                "timeout": {"type": "integer", "description": "Seconds before aborting (default 30)"},
            },
            "required": ["code"],
        },
        func=python_eval,
    ),
    Tool(
        name="run_shell",
        description="Run a shell command in the working directory. Requires --allow-shell.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command line"},
                "timeout": {"type": "integer", "description": "Seconds before aborting (default 30)"},
            },
            "required": ["command"],
        },
        func=run_shell,
    ),
)


class ToolRegistry:
    """Holds the tools that are enabled for this run."""

    def __init__(self, config: Config, confirm: Callable[[str], bool] | None = None) -> None:
        allowed = config.tool_names_allowed()
        self.config = config
        self.context = ToolContext(config=config, confirm=confirm)
        self._tools = {tool.name: tool for tool in ALL_TOOLS if tool.name in allowed}

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [self._tools[name].schema() for name in self.names()]

    def describe(self) -> str:
        return "\n".join(f"  {name:<12} {self._tools[name].description}" for name in self.names())

    def call(self, name: str, arguments: dict[str, Any] | str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            if name in {candidate.name for candidate in ALL_TOOLS}:
                return (
                    f"ERROR: tool {name!r} is disabled for this run. "
                    "Shell access requires restarting with --allow-shell."
                )
            known = ", ".join(self.names()) or "(none)"
            return f"ERROR: unknown tool {name!r}. Available: {known}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return f"ERROR: arguments for {name} are not valid JSON: {exc}"
        if not isinstance(arguments, dict):
            return "ERROR: tool arguments must be a JSON object"

        missing = [key for key in tool.parameters.get("required", []) if key not in arguments]
        if missing:
            return f"ERROR: {name} is missing required argument(s): {', '.join(missing)}"
        try:
            return tool.func(arguments, self.context)
        except ToolError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # noqa: BLE001 - never crash the loop on a bad tool call
            return f"ERROR: {name} failed: {type(exc).__name__}: {exc}"


def quote(path: str) -> str:
    """Small helper used by the CLI for pretty command echoes."""
    return shlex.quote(path)
