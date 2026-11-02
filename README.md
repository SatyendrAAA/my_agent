# my_agent

A small **tool-using CLI agent** in pure Python — no dependencies, no SDK, no
build step. It talks to any OpenAI-compatible `/chat/completions` endpoint
(OpenAI, Groq, Together, Ollama, LM Studio, vLLM, a gateway…) and can also run
completely offline with a deterministic command planner.

```
you> read pyproject.toml and tell me the python version
  -> read_file {"path": "pyproject.toml"}
  <- 1  [build-system]
     ...

The project needs Python >= 3.9 (see requires-python).
```

## Quickstart

```bash
git clone https://github.com/SatyendrAAA/my_agent.git
cd my_agent

# 1. run with no API key at all (deterministic planner + real tools)
python3 -m agent ls

# 2. or point it at a real model
export OPENAI_API_KEY=sk-...
python3 -m agent "list the files here and summarise the README"
```

Install it as a command if you like (`my-agent` instead of `python3 -m agent`):

```bash
pip install -e .        # gives you the `my-agent` entry point
```

## Two modes

| Mode | When | What it does |
| --- | --- | --- |
| **Live** | an API key is present | Full agent loop: model plans, calls tools, observes, answers. |
| **Offline** | no key, or `--offline` | Small built-in planner maps `ls` / `read` / `search` / `python:` / `run` / `write` onto the same tools, so the loop still works without network. |

Offline mode is a convenience for demos and tests — it is not an LLM and only
understands those command shapes. Run `/help`-style commands (`--help`) or just
type something else and it prints what it can do.

## Configuration

Everything is environment variables, overridable by flags:

| Env var | Flag | Default |
| --- | --- | --- |
| `MY_AGENT_API_KEY` / `OPENAI_API_KEY` | `--api-key` | – |
| `MY_AGENT_BASE_URL` / `OPENAI_BASE_URL` | `--base-url` | `https://api.openai.com/v1` |
| `MY_AGENT_MODEL` / `OPENAI_MODEL` | `--model` | `gpt-4o-mini` |
| `MY_AGENT_MAX_STEPS` | `--max-steps` | `8` tool rounds per turn |
| `MY_AGENT_ALLOW_SHELL` | `--allow-shell` | off |
| `MY_AGENT_SESSION_DIR` | – | `~/.my_agent/sessions` |

Local models work too:

```bash
export MY_AGENT_BASE_URL=http://localhost:11434/v1   # Ollama
export MY_AGENT_MODEL=qwen2.5-coder:7b
export MY_AGENT_API_KEY=ollama                        # anything non-empty
python3 -m agent "explain agent/loop.py"
```

## Tools

| Tool | What it does |
| --- | --- |
| `list_dir` | list a directory |
| `read_file` | read text with line numbers (`start_line`, `max_lines`) |
| `write_file` | create/overwrite a file (makes parent dirs) |
| `search_files` | case-insensitive substring or regex search |
| `python_eval` | run a Python snippet in an isolated interpreter |
| `run_shell` | run a shell command — **opt-in** with `--allow-shell` |

Safety rails:

* every path is resolved inside the working directory; `../` and outside
  absolute paths are rejected;
* binary and oversized files are refused instead of dumped into the prompt;
* `run_shell` only exists after `--allow-shell`, and in interactive mode each
  command asks `run this command? [y/N]`;
* subprocesses have timeouts (`python_eval` 30 s, `run_shell` 30 s by default)
  and output is truncated to keep the context small.

## Usage

```bash
python3 -m agent                                   # interactive REPL
python3 -m agent "find every TODO and list the files"   # one-shot
python3 -m agent --resume 20261102-180247          # continue a session
python3 -m agent --continue                        # newest session
python3 -m agent --list-sessions                   # what's saved
python3 -m agent --no-save "quick question"        # don't persist
```

REPL commands:

```
/help              this text
/tools             tools available this run
/history [n]       last n messages (default 6)
/clear             wipe the conversation, keep the session id
/save              save now
/config            effective settings
/exit              quit
```

Sessions are plain JSON (`~/.my_agent/sessions/<id>.json`) — inspect, grep or
delete them by hand. They are written after every turn, so a crash doesn't lose
the transcript.

## How it works

```
cli.py      argparse, REPL, colors, /commands
  └── loop.py       plan → tool calls → observations → answer (max_steps guard)
        ├── llm.py       OpenAI-compatible client over urllib
        ├── offline.py   deterministic planner (same interface as llm.py)
        ├── tools.py     tool registry + path sandboxing
        └── memory.py    session messages + JSON persistence
```

The loop is provider-agnostic: anything that implements
`chat(messages, tools) -> {"role": "assistant", "content": str, "tool_calls": [...]}`
is a valid model, which is exactly how `OfflineModel` and the test fakes plug in
without network access.

## Tests

63 unit tests, stdlib only:

```bash
python3 -m unittest discover -s tests -t .     # or: make test
```

They cover path sandboxing, tool error handling, the multi-round loop, step
limits, session persistence, the CLI (one-shot, offline fallback, `--resume`)
and the HTTP client against a mocked endpoint.

## Known limits

* Streaming and image inputs are not implemented.
* No vector memory — each turn sends the whole session transcript.
* `write_file` writes whole files; there is no surgical `edit_file` yet.
* The offline planner is pattern matching, not reasoning.

## License

MIT
