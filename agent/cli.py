"""Command-line interface for my_agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .config import Config
from .llm import LLMError, OpenAICompatibleModel
from .loop import Agent
from .memory import Session, SessionStore, new_session_id
from .offline import OfflineModel
from .tools import ToolRegistry

BANNER = r"""
  my_agent {version}  -  a tiny tool-using CLI agent
  model: {model}   workdir: {workdir}
  /help for commands, /exit to quit
""".strip("\n")


# --------------------------------------------------------------------------- ui
class UI:
    """Tiny ANSI helper; degrades to plain text when not a TTY or --no-color."""

    def __init__(self, color: bool = True, verbose: bool = False) -> None:
        self.color = color and sys.stdout.isatty()
        self.verbose = verbose

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == "tool_call":
            name = payload.get("name", "?")
            args = payload.get("arguments")
            rendered = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
            if len(rendered) > 160:
                rendered = rendered[:157] + "..."
            print(self.dim(f"  -> {name} {rendered}"))
        elif kind == "tool_result":
            text = (payload.get("result") or "").strip()
            first_line = text.splitlines()[0] if text else "(empty)"
            if len(first_line) > 120:
                first_line = first_line[:117] + "..."
            status = self.red(first_line) if text.startswith("ERROR") else self.dim(first_line)
            print(self.dim("  <- ") + status)
            if self.verbose and len(text.splitlines()) > 1:
                for line in text.splitlines()[1:40]:
                    print(self.dim("     " + line[:160]))


# --------------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="my-agent",
        description="A small tool-using CLI agent (file tools, Python, optional shell).",
    )
    parser.add_argument("prompt", nargs="*", help="one-shot task; omit for interactive mode")
    parser.add_argument("--offline", action="store_true", help="use the deterministic planner, no API key")
    parser.add_argument("--model", help="model name (default: env or gpt-4o-mini)")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL, e.g. http://localhost:11434/v1")
    parser.add_argument("--api-key", help="API key (prefer MY_AGENT_API_KEY / OPENAI_API_KEY)")
    parser.add_argument("--workdir", help="directory the agent may touch (default: current dir)")
    parser.add_argument("--max-steps", type=int, help="max tool rounds per turn (default 8)")
    parser.add_argument("--allow-shell", action="store_true", help="enable the run_shell tool")
    parser.add_argument("--session", help="session id to write to")
    parser.add_argument("--resume", metavar="ID", help="continue an existing session")
    parser.add_argument("--continue", dest="continue_last", action="store_true", help="continue the newest session")
    parser.add_argument("--list-sessions", action="store_true", help="show saved sessions and exit")
    parser.add_argument("--no-save", action="store_true", help="do not persist the session")
    parser.add_argument("--verbose", "-v", action="store_true", help="show full tool output")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    parser.add_argument("--version", action="version", version=f"my_agent {__version__}")
    return parser


def build_config(args: argparse.Namespace) -> Config:
    overrides: dict[str, Any] = {}
    if args.model:
        overrides["model"] = args.model
    if args.base_url:
        overrides["base_url"] = args.base_url
    if args.api_key:
        overrides["api_key"] = args.api_key
    if args.workdir:
        overrides["workdir"] = Path(args.workdir)
    if args.max_steps:
        overrides["max_steps"] = args.max_steps
    if args.allow_shell:
        overrides["allow_shell"] = True

    config = Config(**overrides)
    config.offline = args.offline or not config.has_api_key
    return config


def build_agent(config: Config, ui: UI, session: Session, interactive: bool) -> Agent:
    if config.offline:
        model = OfflineModel()
    else:
        model = OpenAICompatibleModel(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            timeout=config.timeout,
        )

    def confirm(command: str) -> bool:
        if not interactive:
            return True  # explicit --allow-shell opt-in, non-interactive run
        print(ui.yellow(f"  shell> {command}"))
        try:
            answer = input("  run this command? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in {"y", "yes"}

    tools = ToolRegistry(config, confirm=confirm if config.allow_shell else None)
    return Agent(config, model, tools, session, on_event=ui.event)


def load_session(args: argparse.Namespace, config: Config) -> Session:
    store = SessionStore(config.session_dir)
    session_id = args.session
    if args.resume:
        return store.load(args.resume)
    if args.continue_last:
        latest = store.latest()
        if latest is not None:
            return latest
        print("(no saved session yet, starting a new one)")
    return Session(session_id=session_id or new_session_id(), system_prompt=config.system_prompt())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = build_config(args)
    store = SessionStore(config.session_dir)

    if args.list_sessions:
        sessions = store.list()
        if not sessions:
            print(f"no sessions in {config.session_dir}")
            return 0
        for session in sessions:
            stamp = __import__("time").strftime("%Y-%m-%d %H:%M", __import__("time").localtime(session.updated_at))
            turns = sum(1 for m in session.messages if m.get("role") == "user")
            preview = next((m.get("content", "") for m in reversed(session.messages) if m.get("role") == "user"), "")
            preview = preview.replace("\n", " ")[:60]
            print(f"{session.session_id}  {stamp}  {turns:>3} turns  {preview}")
        return 0

    ui = UI(color=not args.no_color, verbose=args.verbose)
    session = load_session(args, config)
    interactive = not args.prompt
    agent = build_agent(config, ui, session, interactive)

    if config.offline and not session.messages:
        ui.event("thinking", {})

    def persist() -> None:
        if args.no_save:
            return
        try:
            store.save(session)
        except OSError as exc:
            print(ui.red(f"could not save session: {exc}"), file=sys.stderr)

    # ------------------------------------------------------------- one-shot mode
    if args.prompt:
        task = " ".join(args.prompt)
        try:
            result = agent.handle(task)
        except LLMError as exc:
            print(ui.red(f"model error: {exc}"), file=sys.stderr)
            return 2
        print()
        print(result.content)
        persist()
        return 0

    # ------------------------------------------------------------ interactive mode
    print(BANNER.format(version=__version__, model=agent.model.name, workdir=config.workdir))
    if config.offline:
        print(ui.yellow("  offline mode: no API key found, using the built-in command planner"))
    if config.allow_shell:
        print(ui.yellow("  shell tool enabled - commands will ask for confirmation"))
    print()

    try:
        while True:
            try:
                line = input(ui.bold("you> ")).strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print(ui.dim("\n(ctrl-c again to quit)"))
                continue

            if not line:
                continue
            if line.startswith("/"):
                if not handle_command(line, agent, config, ui, persist):
                    break
                continue

            try:
                result = agent.handle(line)
            except KeyboardInterrupt:
                print(ui.dim("  (interrupted)"))
                continue
            except LLMError as exc:
                print(ui.red(f"  model error: {exc}"))
                continue

            print()
            print(result.content or "(no answer)")
            print()
            persist()
    except KeyboardInterrupt:
        print()

    persist()
    print(ui.dim(f"session: {session.session_id}  ({len(session.messages)} messages)"))
    return 0


def handle_command(line: str, agent: Agent, config: Config, ui: UI, persist: Any) -> bool:
    """Handle a /slash command. Returns False to exit the REPL."""
    parts = line.split(maxsplit=1)
    command = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""

    if command in {"/exit", "/quit", "/q"}:
        return False

    if command in {"/help", "/h", "/?"}:
        print(
            "\n".join(
                [
                    "  /help              this text",
                    "  /tools             tools available this run",
                    "  /history [n]       show the last n messages (default 6)",
                    "  /clear             wipe the conversation, keep the session id",
                    "  /save              save now",
                    "  /config            show effective settings",
                    "  /exit              quit",
                    "",
                    "Everything else is sent to the model as a normal message.",
                ]
            )
        )
        return True

    if command == "/tools":
        print(ui.cyan("  tools enabled:"))
        print(agent.tools.describe() or "  (none)")
        return True

    if command == "/history":
        limit = int(argument) if argument.isdigit() else 6
        messages = agent.session.messages[-limit:]
        if not messages:
            print("  (empty)")
            return True
        for message in messages:
            role = message.get("role", "?")
            body = (message.get("content") or "").strip().replace("\n", " ")
            if role == "assistant" and message.get("tool_calls"):
                names = ", ".join(
                    (call.get("function") or {}).get("name", "?") for call in message["tool_calls"]
                )
                body = (body + f" [tools: {names}]").strip()
            print(f"  {role:<9} {body[:160]}")
        return True

    if command == "/clear":
        agent.reset()
        print("  conversation cleared")
        return True

    if command == "/save":
        persist()
        print(f"  saved session {agent.session.session_id}")
        return True

    if command == "/config":
        print(f"  model      {agent.model.name}")
        print(f"  base_url   {'(offline)' if config.offline else config.base_url}")
        print(f"  workdir    {config.workdir}")
        print(f"  max_steps  {config.max_steps}")
        print(f"  shell      {'on' if config.allow_shell else 'off'}")
        print(f"  sessions   {config.session_dir}")
        print(f"  session    {agent.session.session_id}")
        return True

    print(ui.yellow(f"  unknown command {command}; try /help"))
    return True


def run() -> None:
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130) from None


if __name__ == "__main__":  # pragma: no cover
    run()
