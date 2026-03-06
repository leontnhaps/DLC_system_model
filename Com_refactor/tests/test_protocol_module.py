"""Tests for infra.protocol constants/builders (stdlib-only)."""

import ast
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

WINDOW_PATH = PROJECT_ROOT / "app" / "window.py"


class _CmdLiteralCollector(ast.NodeVisitor):
    def __init__(self):
        self.commands = set()

    def visit_Dict(self, node):
        for key_node, value_node in zip(node.keys, node.values):
            if (
                isinstance(key_node, ast.Constant)
                and key_node.value == "cmd"
                and isinstance(value_node, ast.Constant)
                and isinstance(value_node.value, str)
            ):
                self.commands.add(value_node.value)
        self.generic_visit(node)


class ProtocolModuleTest(unittest.TestCase):
    def test_command_constants_exist_and_are_strings(self):
        from infra import protocol

        expected_names = [
            "CMD_PREVIEW",
            "CMD_SNAP",
            "CMD_SCAN_RUN",
            "CMD_SCAN_STOP",
            "CMD_MOVE",
            "CMD_LED",
            "CMD_LASER",
            "CMD_IR_CUT",
        ]
        for name in expected_names:
            self.assertTrue(hasattr(protocol, name), f"missing constant: {name}")
            self.assertIsInstance(getattr(protocol, name), str, f"{name} must be str")

    def test_window_cmd_literals_are_covered(self):
        from infra import protocol

        source = WINDOW_PATH.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(WINDOW_PATH))
        collector = _CmdLiteralCollector()
        collector.visit(tree)

        discovered = collector.commands
        self.assertTrue(discovered, "no cmd literals discovered in app/window.py")
        self.assertTrue(hasattr(protocol, "ALL_COMMANDS"))

        missing = discovered - set(protocol.ALL_COMMANDS)
        self.assertFalse(missing, f"protocol.ALL_COMMANDS missing: {sorted(missing)}")

    def test_builder_outputs(self):
        from infra import protocol

        self.assertEqual(
            protocol.build_move_cmd(1.0, 2.0, 100, 1.0),
            {"cmd": "move", "pan": 1.0, "tilt": 2.0, "speed": 100, "acc": 1.0},
        )
        self.assertEqual(
            protocol.build_scan_run_cmd("scan_abc", pan_min=-180, pan_max=180),
            {"cmd": "scan_run", "session": "scan_abc", "pan_min": -180, "pan_max": 180},
        )
        self.assertEqual(
            protocol.build_scan_stop_cmd(),
            {"cmd": "scan_stop"},
        )
        self.assertEqual(
            protocol.build_snap_cmd(640, 480, 95, "a.jpg", shutter_speed=10000, analogue_gain=1.5),
            {
                "cmd": "snap",
                "width": 640,
                "height": 480,
                "quality": 95,
                "save": "a.jpg",
                "shutter_speed": 10000,
                "analogue_gain": 1.5,
            },
        )
        self.assertEqual(
            protocol.build_snap_cmd(320, 240, 90, "b.jpg"),
            {
                "cmd": "snap",
                "width": 320,
                "height": 240,
                "quality": 90,
                "save": "b.jpg",
            },
        )


if __name__ == "__main__":
    unittest.main()
