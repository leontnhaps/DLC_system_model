#!/usr/bin/env python3
"""
Helper methods mixin
Contains utility and delegation methods
"""


class AppHelpersMixin:
    """헬퍼 메서드 믹스인"""
    
    def _restart_preview(self):
        """Preview 재시작 헬퍼"""
        w = self.test_tab.preview_w.get()
        h = self.test_tab.preview_h.get()
        fps = self.test_tab.preview_fps.get()
        q = self.test_tab.preview_q.get()
        # 상태 추적/라벨 갱신이 같이 되도록 toggle_preview 경유
        self.toggle_preview(True, w, h, fps, q)
