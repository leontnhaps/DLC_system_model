"""AST-based tests for ui_components compatibility barrel."""

import ast
from pathlib import Path
import unittest


EXPECTED_IMPORTS = {
    "ui.preview_frame": ["PreviewFrame"],
    "ui.scan_tab": ["ScanTab"],
    "ui.test_settings_tab": ["TestSettingsTab"],
    "ui.pointing_tab": ["PointingTab"],
    "ui.scheduling_tab": ["SchedulingTab"],
}

EXPECTED_EXPORTS = ["PreviewFrame", "ScanTab", "TestSettingsTab", "PointingTab", "SchedulingTab"]


class UIBarrelAstTest(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.ui_components_path = self.project_root / "ui_components.py"
        self.tree = ast.parse(self.ui_components_path.read_text(encoding="utf-8"))

    def test_expected_imports(self):
        imports = {}
        for node in self.tree.body:
            if isinstance(node, ast.ImportFrom):
                imports[node.module] = [alias.name for alias in node.names]
        self.assertEqual(imports, EXPECTED_IMPORTS)

    def test_exports_all_exact(self):
        exports = None
        for node in self.tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        exports = ast.literal_eval(node.value)
        self.assertEqual(exports, EXPECTED_EXPORTS)


if __name__ == "__main__":
    unittest.main()
