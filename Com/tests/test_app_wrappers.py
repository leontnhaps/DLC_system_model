"""Wrapper import tests for app compatibility modules."""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class AppWrapperImportTest(unittest.TestCase):
    def test_app_config_exports(self):
        import app_config

        self.assertTrue(hasattr(app_config, "SERVER_HOST"))
        self.assertTrue(hasattr(app_config, "GUI_CTRL_PORT"))
        self.assertTrue(hasattr(app_config, "GUI_IMG_PORT"))
        self.assertTrue(hasattr(app_config, "SAVE_DIR"))

    def test_config_values_match(self):
        from app.config import SERVER_HOST as a_host
        from app_config import SERVER_HOST as b_host

        self.assertEqual(a_host, b_host)

    def test_app_state_is_same_class(self):
        from app.state import AppState as a_state
        from app_state import AppState as b_state

        self.assertIs(a_state, b_state)


if __name__ == "__main__":
    unittest.main()
