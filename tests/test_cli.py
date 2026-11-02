import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from agent.cli import main
from agent.llm import LLMError, OpenAICompatibleModel


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "hello.txt").write_text("hi there\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_one_shot_offline_read(self):
        code, out, _ = self.run_cli(
            ["read hello.txt", "--offline", "--workdir", str(self.root),
             "--session", "test", "--no-save", "--no-color"]
        )
        self.assertEqual(code, 0)
        self.assertIn("hi there", out)

    def test_one_shot_offline_unknown_prints_help(self):
        code, out, _ = self.run_cli(
            ["do something clever", "--offline", "--session", "t", "--no-save", "--no-color"]
        )
        self.assertEqual(code, 0)
        self.assertIn("Offline mode", out)

    def test_missing_api_key_falls_back_to_offline(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            code, out, _ = self.run_cli(
                ["ls", "--workdir", str(self.root), "--session", "t", "--no-save", "--no-color"]
            )
        self.assertEqual(code, 0)
        self.assertIn("hello.txt", out)

    def test_list_sessions(self):
        with tempfile.TemporaryDirectory():
            code, out, _ = self.run_cli(["--list-sessions", "--no-color"])
            self.assertEqual(code, 0)
            self.assertIsInstance(out, str)
            self.assertIn("no sessions", out)

    def test_session_is_saved_and_resumed(self):
        with tempfile.TemporaryDirectory() as sessions:
            base = ["--offline", "--no-color", "--workdir", str(self.root), "--session", "keep"]
            with mock.patch.dict("os.environ", {"MY_AGENT_SESSION_DIR": sessions}):
                self.run_cli(["read hello.txt", *base])
                code, out, _ = self.run_cli(["--resume", "keep", "--list-sessions"])
            self.assertEqual(code, 0)
            self.assertIn("keep", out)

    def test_model_error_exits_nonzero(self):
        with mock.patch.object(OpenAICompatibleModel, "chat", side_effect=LLMError("boom")):
            code, _, err = self.run_cli(
                ["hello", "--api-key", "k", "--workdir", str(self.root),
                 "--session", "t", "--no-save", "--no-color"]
            )
        self.assertEqual(code, 2)
        self.assertIn("boom", err)

    def test_help_flag(self):
        with self.assertRaises(SystemExit) as ctx:
            self.run_cli(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_version_flag(self):
        with self.assertRaises(SystemExit) as ctx:
            self.run_cli(["--version"])
        self.assertEqual(ctx.exception.code, 0)


class LlmClientTests(unittest.TestCase):
    def test_requires_key_only_for_real_endpoints(self):
        model = OpenAICompatibleModel(model="m", api_key=None, base_url="http://127.0.0.1:1/v1", timeout=0.5)
        with self.assertRaises(LLMError):
            model.chat([{"role": "user", "content": "hi"}], [])

    def test_parses_a_normalised_reply(self):
        model = OpenAICompatibleModel(model="m", api_key="k")
        fake = {"choices": [{"message": {"content": "ok", "tool_calls": None}}]}
        with mock.patch.object(OpenAICompatibleModel, "_post", return_value=fake):
            message = model.chat([{"role": "user", "content": "hi"}], [])
        self.assertEqual(message["content"], "ok")
        self.assertEqual(message["tool_calls"], [])

    def test_empty_choices_raise(self):
        model = OpenAICompatibleModel(model="m", api_key="k")
        with mock.patch.object(OpenAICompatibleModel, "_post", return_value={"choices": []}):
            with self.assertRaises(LLMError):
                model.chat([], [])


if __name__ == "__main__":
    unittest.main()
