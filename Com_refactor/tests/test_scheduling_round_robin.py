"""Tests for scheduling.round_robin module (stdlib-only)."""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class SchedulingRoundRobinTest(unittest.TestCase):
    def test_import(self):
        from scheduling.round_robin import RoundRobinScheduler

        self.assertTrue(callable(RoundRobinScheduler))

    def test_empty_candidates(self):
        from scheduling.round_robin import RoundRobinScheduler

        scheduler = RoundRobinScheduler()
        selected, state = scheduler.select_next([])
        self.assertIsNone(selected)
        self.assertEqual(state, {"index": 0})

    def test_rotation_is_deterministic(self):
        from scheduling.round_robin import RoundRobinScheduler

        scheduler = RoundRobinScheduler()
        state = None
        result = []
        candidates = ["A", "B", "C"]

        for _ in range(5):
            selected, state = scheduler.select_next(candidates, state=state)
            result.append(selected)

        self.assertEqual(result, ["A", "B", "C", "A", "B"])
        self.assertEqual(state, {"index": 2})


if __name__ == "__main__":
    unittest.main()
