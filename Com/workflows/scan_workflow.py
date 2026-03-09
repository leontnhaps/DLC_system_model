"""Minimal scan workflow scaffold delegating to ScanController."""


class ScanWorkflow:
    """Thin orchestration wrapper around an existing ScanController."""

    def __init__(self, scan_controller):
        self.scan_controller = scan_controller

    def start_session(self, yolo_weights_path=None):
        return self.scan_controller.start_session(yolo_weights_path=yolo_weights_path)

    def stop_session(self):
        return self.scan_controller.stop_session()

    def is_active(self):
        return self.scan_controller.is_active()

    def save_image(self, name, data):
        return self.scan_controller.save_image(name, data)

    def update_progress(self, done, total):
        return self.scan_controller.update_progress(done, total)

    def get_progress(self):
        return self.scan_controller.get_progress()

    def get_session_name(self):
        return self.scan_controller.get_session_name()


__all__ = ["ScanWorkflow"]
