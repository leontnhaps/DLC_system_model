"""Wrapper import tests for utils compatibility modules."""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class UtilsWrapperImportTest(unittest.TestCase):
    def test_root_naming_import(self):
        import naming

        self.assertTrue(hasattr(naming, "parse_image_name"))

    def test_wrapper_and_impl_are_same_function(self):
        from naming import parse_image_name
        from utils.naming import parse_image_name as parse2

        self.assertIs(parse_image_name, parse2)


if __name__ == "__main__":
    unittest.main()
