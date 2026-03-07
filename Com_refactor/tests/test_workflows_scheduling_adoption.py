"""AST checks for SchedulingWorkflow adoption wiring (stdlib-only)."""

import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDOW_PATH = PROJECT_ROOT / "app" / "window.py"


class SchedulingWorkflowAdoptionAstTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = WINDOW_PATH.read_text(encoding="utf-8-sig")
        cls.tree = ast.parse(source, filename=str(WINDOW_PATH))

    def test_window_imports_schedulingworkflow(self):
        imported = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                if node.module in ("workflows", "workflows.scheduling_workflow"):
                    if any(alias.name == "SchedulingWorkflow" for alias in node.names):
                        imported = True
                        break
        self.assertTrue(imported, "window.py에서 SchedulingWorkflow import를 찾지 못했습니다.")

    def test_window_assigns_self_scheduling_workflow(self):
        found = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and target.attr == "scheduling_workflow"
                    ):
                        found = True
                        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                            self.assertEqual(node.value.func.id, "SchedulingWorkflow")
                        break
        self.assertTrue(found, "self.scheduling_workflow 할당을 찾지 못했습니다.")

    def test_window_has_scheduling_workflow_reference_path(self):
        has_backend_method = False
        has_attr_use = False

        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_get_scheduling_backend":
                has_backend_method = True

            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr == "scheduling_workflow"
            ):
                has_attr_use = True

        self.assertTrue(has_backend_method or has_attr_use)


if __name__ == "__main__":
    unittest.main()
