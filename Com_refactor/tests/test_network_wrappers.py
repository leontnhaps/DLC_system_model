"""Wrapper import tests for network compatibility module."""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class NetworkWrapperImportTest(unittest.TestCase):
    def test_network_wrapper_identity(self):
        import network
        from infra import network_client

        self.assertIs(network.GuiCtrlClient, network_client.GuiCtrlClient)
        self.assertIs(network.GuiImgClient, network_client.GuiImgClient)
        self.assertIs(network.ui_q, network_client.ui_q)
        self.assertTrue(hasattr(network, "pop_latest_preview"))
        self.assertTrue(hasattr(network_client, "pop_latest_preview"))
        self.assertIs(network.pop_latest_preview, network_client.pop_latest_preview)


if __name__ == "__main__":
    unittest.main()
