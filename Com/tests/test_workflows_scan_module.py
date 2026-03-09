"""Tests for workflows.scan_workflow (stdlib-only)."""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _FakeController:
    def __init__(self):
        self.calls = []

    def start_session(self, yolo_weights_path=None):
        self.calls.append(("start_session", yolo_weights_path))
        return "scan_20260101_000000"

    def stop_session(self):
        self.calls.append(("stop_session",))
        return {"session": "scan_20260101_000000", "done": 1, "total": 1}

    def is_active(self):
        self.calls.append(("is_active",))
        return True

    def save_image(self, name, data):
        self.calls.append(("save_image", name, data))
        return f"/fake/{name}"

    def update_progress(self, done, total):
        self.calls.append(("update_progress", done, total))
        return None

    def get_progress(self):
        self.calls.append(("get_progress",))
        return (3, 10)

    def get_session_name(self):
        self.calls.append(("get_session_name",))
        return "scan_20260101_000000"


class ScanWorkflowModuleTest(unittest.TestCase):
    def test_import(self):
        from workflows.scan_workflow import ScanWorkflow

        self.assertTrue(callable(ScanWorkflow))

    def test_delegation_core(self):
        from workflows.scan_workflow import ScanWorkflow

        fake = _FakeController()
        workflow = ScanWorkflow(fake)

        self.assertTrue(workflow.is_active())
        self.assertEqual(workflow.save_image("a.jpg", b"data"), "/fake/a.jpg")

        self.assertIn(("is_active",), fake.calls)
        self.assertIn(("save_image", "a.jpg", b"data"), fake.calls)

    def test_delegation_extra_methods(self):
        from workflows.scan_workflow import ScanWorkflow

        fake = _FakeController()
        workflow = ScanWorkflow(fake)

        self.assertEqual(workflow.start_session("weights.pt"), "scan_20260101_000000")
        self.assertEqual(
            workflow.stop_session(),
            {"session": "scan_20260101_000000", "done": 1, "total": 1},
        )
        workflow.update_progress(3, 10)
        self.assertEqual(workflow.get_progress(), (3, 10))
        self.assertEqual(workflow.get_session_name(), "scan_20260101_000000")

        self.assertIn(("start_session", "weights.pt"), fake.calls)
        self.assertIn(("stop_session",), fake.calls)
        self.assertIn(("update_progress", 3, 10), fake.calls)
        self.assertIn(("get_progress",), fake.calls)
        self.assertIn(("get_session_name",), fake.calls)


if __name__ == "__main__":
    unittest.main()
