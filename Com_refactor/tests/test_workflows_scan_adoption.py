"""AST checks for ScanWorkflow adoption wiring (stdlib-only)."""

import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDOW_PATH = PROJECT_ROOT / "app" / "window.py"
EVENT_HANDLERS_PATH = PROJECT_ROOT / "app" / "event_handlers.py"


class ScanWorkflowAdoptionAstTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.window_source = WINDOW_PATH.read_text(encoding="utf-8-sig")
        cls.window_tree = ast.parse(cls.window_source, filename=str(WINDOW_PATH))

        cls.handlers_source = EVENT_HANDLERS_PATH.read_text(encoding="utf-8-sig")
        cls.handlers_tree = ast.parse(cls.handlers_source, filename=str(EVENT_HANDLERS_PATH))

    def test_window_references_scanworkflow(self):
        imported = False
        for node in ast.walk(self.window_tree):
            if isinstance(node, ast.ImportFrom):
                if node.module in ("workflows", "workflows.scan_workflow"):
                    if any(alias.name == "ScanWorkflow" for alias in node.names):
                        imported = True
                        break
        self.assertTrue(imported, "window.py에서 ScanWorkflow import를 찾지 못했습니다.")

    def test_window_assigns_self_scan_workflow(self):
        found_assignment = False
        for node in ast.walk(self.window_tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr == "scan_workflow"
                    ):
                        found_assignment = True
                        if isinstance(node.value, ast.Call):
                            if isinstance(node.value.func, ast.Name):
                                self.assertEqual(node.value.func.id, "ScanWorkflow")
                        break
        self.assertTrue(found_assignment, "self.scan_workflow 할당을 찾지 못했습니다.")

    def test_event_handlers_has_scan_workflow_fallback(self):
        get_backend = None
        for node in self.handlers_tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "EventHandlersMixin":
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == "_get_scan_backend":
                        get_backend = child
                        break
        self.assertIsNotNone(get_backend, "_get_scan_backend 메서드를 찾지 못했습니다.")

        attrs = set()
        has_hasattr_scan_workflow = False
        for node in ast.walk(get_backend):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "self" and node.attr in ("scan_workflow", "scan_ctrl"):
                    attrs.add(node.attr)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "hasattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "scan_workflow"
            ):
                has_hasattr_scan_workflow = True

        self.assertIn("scan_workflow", attrs)
        self.assertIn("scan_ctrl", attrs)
        self.assertTrue(has_hasattr_scan_workflow, "scan_workflow 존재 검사(hasattr)를 찾지 못했습니다.")


if __name__ == "__main__":
    unittest.main()
