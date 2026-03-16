"""Wrapper import tests for infra compatibility modules."""

import importlib
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class InfraWrapperImportTest(unittest.TestCase):
    def test_root_wrappers_export_symbols(self):
        infra_event_bus = importlib.import_module("infra_event_bus")
        image_router = importlib.import_module("image_router")

        self.assertTrue(hasattr(infra_event_bus, "EventBus"))
        self.assertTrue(hasattr(image_router, "route_saved_image"))

    def test_infra_modules_import(self):
        from infra.event_bus import EventBus
        from infra.image_router import route_saved_image

        self.assertIsNotNone(EventBus)
        self.assertIsNotNone(route_saved_image)


if __name__ == "__main__":
    unittest.main()
