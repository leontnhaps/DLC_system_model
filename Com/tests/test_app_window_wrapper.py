"""AST checks for Com_main compatibility wrapper."""

import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COM_MAIN_PATH = PROJECT_ROOT / "Com_main.py"


class AppWindowWrapperAstTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = COM_MAIN_PATH.read_text(encoding="utf-8-sig")
        cls.tree = ast.parse(cls.source, filename=str(COM_MAIN_PATH))

    def test_imports_from_app_window(self):
        imports = [
            node
            for node in self.tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "app.window"
        ]
        self.assertTrue(imports, "app.window에서 import 하는 구문이 없습니다.")
        imported_names = {alias.name for node in imports for alias in node.names}
        self.assertIn("ComApp", imported_names)
        self.assertIn("main", imported_names)

    def test_all_exports(self):
        all_assigns = [
            node
            for node in self.tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
        ]
        self.assertTrue(all_assigns, "__all__ 할당이 없습니다.")
        values = []
        for node in all_assigns:
            if isinstance(node.value, (ast.List, ast.Tuple)):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        values.append(elt.value)
        self.assertIn("ComApp", values)
        self.assertIn("main", values)

    def test_main_guard_calls_main(self):
        guards = [
            node
            for node in self.tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value == "__main__"
        ]
        self.assertTrue(guards, "__name__ == '__main__' 가드가 없습니다.")

        has_main_call = False
        for guard in guards:
            for stmt in guard.body:
                if (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Name)
                    and stmt.value.func.id == "main"
                ):
                    has_main_call = True
                    break
            if has_main_call:
                break

        self.assertTrue(has_main_call, "main() 호출이 __main__ 가드 안에 없습니다.")


if __name__ == "__main__":
    unittest.main()
