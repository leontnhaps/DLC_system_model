"""Tests for utils.threading helpers (stdlib-only)."""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _FakeRoot:
    def __init__(self):
        self.delays = []

    def after(self, delay, callback):
        self.delays.append(delay)
        callback()
        return "after-token"


class UtilsThreadingTest(unittest.TestCase):
    def test_call_on_ui_thread(self):
        from utils.threading import call_on_ui_thread

        fake_root = _FakeRoot()
        called = {}

        def _target(a, b, x=None):
            called["args"] = (a, b)
            called["kwargs"] = {"x": x}

        token = call_on_ui_thread(fake_root, _target, 1, 2, x=3)

        self.assertEqual(fake_root.delays, [0])
        self.assertEqual(called.get("args"), (1, 2))
        self.assertEqual(called.get("kwargs"), {"x": 3})
        self.assertEqual(token, "after-token")


if __name__ == "__main__":
    unittest.main()
