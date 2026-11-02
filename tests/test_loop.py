import json
import tempfile
import unittest
from pathlib import Path

from agent.config import Config
from agent.loop import Agent, _parse_call
from agent.memory import Session
from agent.tools import ToolRegistry


class FakeModel:
    """Replays a scripted list of assistant messages and records what it saw."""

    name = "fake"

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.script:
            return {"role": "assistant", "content": "done", "tool_calls": []}
        return self.script.pop(0)


def tool_call(name, arguments, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data.txt").write_text("payload\n", encoding="utf-8")
        self.config = Config(workdir=self.root, max_steps=4)
        self.events = []

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, script, session=None):
        model = FakeModel(script)
        tools = ToolRegistry(self.config)
        session = session or Session(session_id="t", system_prompt="sys")
        agent = Agent(self.config, model, tools, session, on_event=lambda k, p: self.events.append((k, p)))
        return agent, model

    def test_plain_answer_appends_one_message(self):
        agent, model = self.build([{"role": "assistant", "content": "hello", "tool_calls": []}])
        result = agent.handle("hi")
        self.assertEqual(result.content, "hello")
        self.assertEqual(result.steps, 1)
        self.assertEqual([m["role"] for m in agent.session.messages], ["user", "assistant"])
        self.assertEqual(model.calls[0]["messages"][0]["role"], "system")

    def test_tool_call_round_trip(self):
        agent, model = self.build(
            [
                {"role": "assistant", "content": "", "tool_calls": [tool_call("read_file", {"path": "data.txt"})]},
                {"role": "assistant", "content": "It says payload.", "tool_calls": []},
            ]
        )
        result = agent.handle("what is in data.txt?")
        self.assertEqual(result.content, "It says payload.")
        self.assertEqual(result.tool_names, ["read_file"])
        roles = [m["role"] for m in agent.session.messages]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant"])
        self.assertIn("payload", agent.session.messages[2]["content"])
        # the tool result is visible to the model on the second round
        self.assertEqual(model.calls[1]["messages"][-1]["role"], "tool")

    def test_tool_errors_are_fed_back_not_raised(self):
        agent, _ = self.build(
            [
                {"role": "assistant", "content": "", "tool_calls": [tool_call("read_file", {"path": "ghost.txt"})]},
                {"role": "assistant", "content": "File missing.", "tool_calls": []},
            ]
        )
        result = agent.handle("read ghost.txt")
        self.assertEqual(result.content, "File missing.")
        self.assertIn("ERROR", agent.session.messages[2]["content"])

    def test_multiple_tool_calls_in_one_round(self):
        agent, _ = self.build(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        tool_call("read_file", {"path": "data.txt"}, "c1"),
                        tool_call("list_dir", {}, "c2"),
                    ],
                },
                {"role": "assistant", "content": "ok", "tool_calls": []},
            ]
        )
        result = agent.handle("inspect")
        self.assertEqual(result.tool_names, ["read_file", "list_dir"])
        tool_messages = [m for m in agent.session.messages if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 2)
        self.assertEqual(tool_messages[0]["tool_call_id"], "c1")

    def test_max_steps_stops_the_loop(self):
        script = [
            {"role": "assistant", "content": "", "tool_calls": [tool_call("list_dir", {})]} for _ in range(10)
        ]
        agent, model = self.build(script)
        result = agent.handle("loop forever")
        self.assertTrue(result.truncated)
        self.assertEqual(len(model.calls), self.config.max_steps)
        self.assertIn("stopped after", result.content)

    def test_events_are_emitted(self):
        agent, _ = self.build(
            [
                {"role": "assistant", "content": "", "tool_calls": [tool_call("list_dir", {})]},
                {"role": "assistant", "content": "done", "tool_calls": []},
            ]
        )
        agent.handle("list")
        kinds = [kind for kind, _ in self.events]
        for expected in ("turn_start", "thinking", "tool_call", "tool_result", "assistant"):
            self.assertIn(expected, kinds)

    def test_empty_input_is_ignored(self):
        agent, model = self.build([])
        result = agent.handle("   ")
        self.assertEqual(result.content, "(empty message)")
        self.assertEqual(model.calls, [])

    def test_reset_restores_system_prompt(self):
        agent, _ = self.build([{"role": "assistant", "content": "hi", "tool_calls": []}])
        agent.handle("hello")
        agent.reset()
        self.assertEqual(agent.session.messages, [])
        self.assertIn(str(self.root), agent.session.system_prompt)


class ParseCallTests(unittest.TestCase):
    def test_json_string_arguments(self):
        name, args, call_id = _parse_call(tool_call("read_file", {"path": "a"}), 0)
        self.assertEqual((name, args["path"], call_id), ("read_file", "a", "call_1"))

    def test_empty_string_arguments(self):
        call = {"id": "x", "function": {"name": "list_dir", "arguments": ""}}
        self.assertEqual(_parse_call(call, 0)[1], {})

    def test_malformed_arguments_survive(self):
        call = {"id": "x", "function": {"name": "read_file", "arguments": "{oops"}}
        self.assertIsInstance(_parse_call(call, 0)[1], str)

    def test_missing_id_gets_index(self):
        self.assertEqual(_parse_call({"function": {"name": "list_dir"}}, 3)[2], "call_3")


if __name__ == "__main__":
    unittest.main()
