import tempfile
import unittest
from pathlib import Path

from agent.memory import Session, SessionStore


class SessionTests(unittest.TestCase):
    def test_history_puts_system_prompt_first(self):
        session = Session(session_id="s1", system_prompt="be nice")
        session.append({"role": "user", "content": "hi"})
        history = session.history()
        self.assertEqual(history[0]["role"], "system")
        self.assertEqual(history[1]["content"], "hi")

    def test_clear_keeps_id(self):
        session = Session(session_id="s1", system_prompt="x")
        session.append({"role": "user", "content": "hi"})
        session.clear()
        self.assertEqual(session.messages, [])
        self.assertEqual(session.session_id, "s1")

    def test_roundtrip_through_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            session = Session(session_id="abc", system_prompt="sys")
            session.append({"role": "user", "content": "hello"})
            path = store.save(session)
            self.assertTrue(path.exists())

            loaded = store.load("abc")
            self.assertEqual(loaded.session_id, "abc")
            self.assertEqual(loaded.system_prompt, "sys")
            self.assertEqual(loaded.messages[0]["content"], "hello")

    def test_latest_returns_most_recent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            older = Session(session_id="old", system_prompt="s")
            newer = Session(session_id="new", system_prompt="s")
            store.save(older)
            store.save(newer)
            newer.append({"role": "user", "content": "later"})
            store.save(newer)
            self.assertEqual(store.latest().session_id, "new")

    def test_list_sessions_is_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            for name in ("a", "b", "c"):
                store.save(Session(session_id=name, system_prompt="s"))
            self.assertEqual(len(store.list()), 3)

    def test_invalid_session_id_rejected(self):
        store = SessionStore(Path("/tmp"))
        with self.assertRaises(ValueError):
            store.path_for("../../etc/passwd")

    def test_sanitises_stray_characters(self):
        store = SessionStore(Path("/tmp"))
        self.assertEqual(store.path_for("run 1!").name, "run1.json")

    def test_missing_session_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                SessionStore(Path(tmp)).load("nope")


if __name__ == "__main__":
    unittest.main()
