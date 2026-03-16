"""Wrapper import tests for event_handlers compatibility module."""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class EventHandlersWrapperImportTest(unittest.TestCase):
    def test_wrapper_and_package_identity(self):
        import event_handlers
        from app import event_handlers as app_event_handlers_pkg

        self.assertTrue(hasattr(event_handlers, "EventHandlersMixin"))
        self.assertTrue(hasattr(app_event_handlers_pkg, "EventHandlersMixin"))
        self.assertIs(event_handlers.EventHandlersMixin, app_event_handlers_pkg.EventHandlersMixin)

    def test_wrapper_all_exports(self):
        import event_handlers
        from app import event_handlers as app_event_handlers_pkg

        for name in getattr(event_handlers, "__all__", []):
            self.assertTrue(hasattr(event_handlers, name))
            self.assertTrue(hasattr(app_event_handlers_pkg, name))


if __name__ == "__main__":
    unittest.main()
