import json
import tempfile
import unittest
from pathlib import Path

from agent.config import Config
from agent.loop import Agent
from agent.memory import Session
from agent.offline import OfflineModel
from agent.tools import ToolRegistry


def plan(text):
    model = OfflineModel()
    message = model.chat([{"role": "user", "content": text}], [])
    calls = message.get("tool_calls") or []
    if not calls:
        return None, message["content"]
    function = calls[0]["function"]
    return function["name"], json.loads(function["arguments"])


class OfflinePlannerTests(unittest.TestCase):
    def test_list(self):
        for text in ("ls", "list files", "dir"):
            self.assertEqual(plan(text)[0], "list_dir")

    def test_read(self):
        self.assertEqual(plan("read README.md")[1]["path"], "README.md")
        self.assertEqual(plan("padho notes.txt")[1]["path"], "notes.txt")

    def test_read_strips_quotes(self):
        self.assertEqual(plan("read 'my file.txt'")[1]["path"], "my file.txt")

    def test_search(self):
        self.assertEqual(plan("grep TODO")[1]["query"], "TODO")
        self.assertEqual(plan("dhundo def main")[1]["query"], "def main")

    def test_python(self):
        self.assertEqual(plan("python: print(1+1)")[1]["code"], "print(1+1)")

    def test_write(self):
        name, args = plan("write a.txt :: hello")
        self.assertEqual((name, args["path"], args["content"]), ("write_file", "a.txt", "hello"))

    def test_shell(self):
        self.assertEqual(plan("run echo hi")[1]["command"], "echo hi")

    def test_unknown_returns_help(self):
        name, content = plan("please refactor my whole codebase")
        self.assertIsNone(name)
        self.assertIn("Offline mode", content)

    def test_summarises_previous_tool_output(self):
        model = OfflineModel()
        messages = [
            {"role": "user", "content": "ls"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "tool", "tool_call_id": "1", "name": "list_dir", "content": "file  README.md"},
        ]
        self.assertEqual(model.chat(messages, [])["content"], "file  README.md")


class OfflineAgentEndToEndTests(unittest.TestCase):
    def test_reads_a_real_file_without_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.txt").write_text("hello world\n", encoding="utf-8")
            config = Config(workdir=root)
            agent = Agent(
                config,
                OfflineModel(),
                ToolRegistry(config),
                Session(session_id="off", system_prompt=config.system_prompt()),
            )
            result = agent.handle("read notes.txt")
            self.assertIn("hello world", result.content)
            self.assertEqual(result.tool_names, ["read_file"])

    def test_python_tool_through_the_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Config(workdir=Path(tmp))
            agent = Agent(
                config,
                OfflineModel(),
                ToolRegistry(config),
                Session(session_id="off", system_prompt=config.system_prompt()),
            )
            self.assertEqual(agent.handle("python: print(2**10)").content.strip(), "1024")


if __name__ == "__main__":
    unittest.main()
