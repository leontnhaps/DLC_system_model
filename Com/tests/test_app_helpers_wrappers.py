"""Wrapper import tests for app_helpers compatibility module."""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class AppHelpersWrapperImportTest(unittest.TestCase):
    def test_wrapper_and_package_identity(self):
        import app_helpers
        from app import helpers as app_helpers_pkg

        self.assertTrue(hasattr(app_helpers, "AppHelpersMixin"))
        self.assertTrue(hasattr(app_helpers_pkg, "AppHelpersMixin"))
        self.assertIs(app_helpers.AppHelpersMixin, app_helpers_pkg.AppHelpersMixin)

    def test_wrapper_all_exports(self):
        import app_helpers
        from app import helpers as app_helpers_pkg

        for name in getattr(app_helpers, "__all__", []):
            self.assertTrue(hasattr(app_helpers, name))
            self.assertTrue(hasattr(app_helpers_pkg, name))


if __name__ == "__main__":
    unittest.main()
