"""Tests for scheduling.base module (stdlib-only)."""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class SchedulingBaseModuleTest(unittest.TestCase):
    def test_base_interface_import(self):
        from scheduling.base import SchedulingAlgorithm

        self.assertTrue(hasattr(SchedulingAlgorithm, "select_next"))
        self.assertTrue(callable(SchedulingAlgorithm.select_next))


if __name__ == "__main__":
    unittest.main()
