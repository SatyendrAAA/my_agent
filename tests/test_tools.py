import tempfile
import unittest
from pathlib import Path

from agent.config import Config
from agent.tools import ToolRegistry


def make_registry(**kwargs):
    tmp = tempfile.TemporaryDirectory()
    config = Config(workdir=Path(tmp.name), **kwargs)
    return tmp, ToolRegistry(config, confirm=lambda _cmd: True)


class ToolRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp, self.registry = make_registry()
        self.root = Path(self.tmp.name)
        (self.root / "hello.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "mod.py").write_text("needle = 1\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_shell_tool_hidden_by_default(self):
        self.assertNotIn("run_shell", self.registry.names())
        self.assertIn("read_file", self.registry.names())

    def test_shell_tool_available_when_enabled(self):
        tmp, registry = make_registry(allow_shell=True)
        try:
            self.assertIn("run_shell", registry.names())
        finally:
            tmp.cleanup()

    def test_read_file_numbers_lines(self):
        out = self.registry.call("read_file", {"path": "hello.txt"})
        self.assertIn("1  alpha", out)
        self.assertIn("3  gamma", out)

    def test_read_file_window(self):
        out = self.registry.call("read_file", {"path": "hello.txt", "start_line": 2, "max_lines": 1})
        self.assertIn("2  beta", out)
        self.assertNotIn("alpha", out)

    def test_list_dir(self):
        out = self.registry.call("list_dir", {})
        self.assertIn("hello.txt", out)
        self.assertIn("pkg", out)

    def test_write_file_creates_directories(self):
        out = self.registry.call("write_file", {"path": "a/b/c.txt", "content": "hi\n"})
        self.assertIn("created", out)
        self.assertEqual((self.root / "a" / "b" / "c.txt").read_text(encoding="utf-8"), "hi\n")

    def test_write_file_overwrites(self):
        self.registry.call("write_file", {"path": "hello.txt", "content": "new\n"})
        self.assertIn("updated", self.registry.call("write_file", {"path": "hello.txt", "content": "newer\n"}))
        self.assertEqual((self.root / "hello.txt").read_text(encoding="utf-8"), "newer\n")

    def test_search_files_finds_nested_match(self):
        out = self.registry.call("search_files", {"query": "needle"})
        self.assertIn("pkg/mod.py:1", out)

    def test_search_files_regex(self):
        self.assertIn("needle", self.registry.call("search_files", {"query": r"need\w+", "regex": True}))

    def test_python_eval(self):
        out = self.registry.call("python_eval", {"code": "print(6 * 7)"})
        self.assertEqual(out.strip(), "42")

    def test_python_eval_reports_errors(self):
        out = self.registry.call("python_eval", {"code": "raise SystemExit(3)"})
        self.assertTrue(out.strip())

    def test_path_escape_is_blocked(self):
        out = self.registry.call("read_file", {"path": "../../etc/passwd"})
        self.assertTrue(out.startswith("ERROR"))
        self.assertIn("escapes", out)

    def test_absolute_path_outside_workdir_blocked(self):
        out = self.registry.call("write_file", {"path": "/tmp/evil.txt", "content": "x"})
        self.assertTrue(out.startswith("ERROR"))

    def test_missing_argument_is_reported(self):
        out = self.registry.call("read_file", {})
        self.assertIn("missing required argument", out)

    def test_unknown_tool_is_reported(self):
        out = self.registry.call("nope", {})
        self.assertIn("unknown tool", out)

    def test_bad_json_arguments_are_reported(self):
        out = self.registry.call("read_file", "{not json")
        self.assertIn("not valid JSON", out)

    def test_missing_file_is_reported(self):
        out = self.registry.call("read_file", {"path": "nope.txt"})
        self.assertTrue(out.startswith("ERROR"))
        self.assertIn("no such file", out)

    def test_shell_requires_flag(self):
        out = self.registry.call("run_shell", {"command": "echo hi"})
        self.assertIn("disabled", out)

    def test_shell_runs_when_allowed(self):
        tmp, registry = make_registry(allow_shell=True)
        try:
            out = registry.call("run_shell", {"command": "echo hi"})
            self.assertEqual(out.strip(), "hi")
        finally:
            tmp.cleanup()

    def test_shell_respects_confirmation(self):
        tmp, _ = make_registry(allow_shell=True)
        try:
            config = Config(workdir=Path(tmp.name), allow_shell=True)
            registry = ToolRegistry(config, confirm=lambda _cmd: False)
            out = registry.call("run_shell", {"command": "echo hi"})
            self.assertIn("rejected", out)
        finally:
            tmp.cleanup()

    def test_schemas_are_valid_function_definitions(self):
        for schema in self.registry.schemas():
            self.assertEqual(schema["type"], "function")
            self.assertIn("name", schema["function"])
            self.assertEqual(schema["function"]["parameters"]["type"], "object")


if __name__ == "__main__":
    unittest.main()
