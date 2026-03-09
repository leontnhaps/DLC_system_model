"""Tests for workflows.scheduling_workflow module (stdlib-only)."""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _FakeScheduler:
    def __init__(self):
        self.calls = []

    def select_next(self, candidates, state=None):
        self.calls.append((list(candidates), state))
        return "picked", {"index": 123}


class WorkflowsSchedulingModuleTest(unittest.TestCase):
    def test_import(self):
        from workflows.scheduling_workflow import SchedulingWorkflow

        self.assertTrue(callable(SchedulingWorkflow))

    def test_choose_next_delegates_to_scheduler(self):
        from workflows.scheduling_workflow import SchedulingWorkflow

        fake = _FakeScheduler()
        workflow = SchedulingWorkflow(scheduler=fake)
        selected, state = workflow.choose_next(["x", "y"], state={"index": 1})

        self.assertEqual(selected, "picked")
        self.assertEqual(state, {"index": 123})
        self.assertEqual(fake.calls, [(["x", "y"], {"index": 1})])

    def test_set_scheduler(self):
        from workflows.scheduling_workflow import SchedulingWorkflow

        wf = SchedulingWorkflow()
        fake = _FakeScheduler()
        returned = wf.set_scheduler(fake)

        self.assertIs(returned, fake)
        self.assertIs(wf.scheduler, fake)


if __name__ == "__main__":
    unittest.main()
