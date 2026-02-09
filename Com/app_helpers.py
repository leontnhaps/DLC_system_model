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
        
        self.ctrl.send({
            "cmd": "preview",
            "enable": True,
            "width": w,
            "height": h,
            "fps": fps,
            "quality": q
        })
