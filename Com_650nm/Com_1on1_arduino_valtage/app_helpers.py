#!/usr/bin/env python3
"""
Helper methods mixin for App class
Contains delegation and utility methods
"""

import numpy as np
from network import ui_q

# Constants imported from main
CENTERING_GAIN_PAN = 0.02
CENTERING_GAIN_TILT = 0.02


class AppHelpersMixin:
    """Helper methods for the main App class"""
    
    # ========== Command Helpers ==========
    
    def _send_snap_cmd(self, save_name: str):
        """Snap 명령 전송 헬퍼"""
        self.ctrl.send({
            "cmd": "snap",
            "width": self.width.get(),
            "height": self.height.get(),
            "quality": self.quality.get(),
            "save": save_name,
            "ud_save": self.ud_save_copy.get()
        })

    def _get_yolo_model(self):
        """YOLO 모델 캐싱 - delegates to YOLOProcessor"""
        wpath = self.yolo_wpath.get().strip()
        if not wpath:
            return None
        return self.yolo_processor.get_model(wpath)

    def _undistort_pair(self, img_on, img_off):
        """이미지 쌍 Undistort 헬퍼 - delegates to ImageProcessor"""
        self.image_processor.alpha = float(self.ud_alpha.get())
        return self.image_processor.undistort_pair(img_on, img_off, use_torch=True)

    def _undistort_bgr(self, bgr: np.ndarray) -> np.ndarray:
        """Undistort BGR image - delegates to ImageProcessor"""
        # Update alpha before undistortion
        self.image_processor.alpha = float(self.ud_alpha.get())
        # Use Torch acceleration if available for best performance
        return self.image_processor.undistort(bgr, use_torch=True)

    def _calculate_angle_delta(self, err_x: float, err_y: float, 
                               k_pan: float = CENTERING_GAIN_PAN, k_tilt: float = CENTERING_GAIN_TILT,
                               force_max_step: float = None):
        """픽셀 오차 → 각도 변환 (클램핑 포함)"""
        d_pan = err_x * k_pan
        d_tilt = -err_y * k_tilt
        if force_max_step is not None:
            max_step = force_max_step
        else: 
            max_step = self.pointing_max_step.get()
        
        d_pan = max(min(d_pan, max_step), -max_step)
        d_tilt = max(min(d_tilt, max_step), -max_step)
        return d_pan, d_tilt

    def _get_device(self):
        """YOLO/Torch 디바이스 반환 - delegates to YOLOProcessor"""
        return self.yolo_processor.get_device()

    # ========== GUI Helpers ==========
    
    def resume_preview(self):
        if self.preview_enable.get():
            self.ctrl.send({
                "cmd":"preview", "enable": True,
                "width":  self.preview_w.get(),
                "height": self.preview_h.get(),
                "fps":    self.preview_fps.get(),
                "quality":self.preview_q.get(),
            })

    def _row(self,parent,r,label,v1,v2,v3=None,caps=("min","max","step")):
        from tkinter import Label, ttk
        Label(parent,text=label).grid(row=r,column=0,sticky="w",padx=4,pady=2)
        ttk.Entry(parent,width=8,textvariable=v1).grid(row=r,column=1,sticky="w",padx=4)
        ttk.Entry(parent,width=8,textvariable=v2).grid(row=r,column=2,sticky="w",padx=4)
        if v3 is not None:
            ttk.Entry(parent,width=8,textvariable=v3).grid(row=r,column=3,sticky="w",padx=4)
    
    def _entry(self,parent,r,label,var):
        from tkinter import Label, ttk
        Label(parent,text=label).grid(row=r,column=0,sticky="w",padx=4,pady=2)
        ttk.Entry(parent,width=8,textvariable=var).grid(row=r,column=1,sticky="w",padx=4)
    
    def _slider(self,parent,r,label,a,b,var,res):
        from tkinter import Label, Scale, HORIZONTAL
        Label(parent,text=label).grid(row=r,column=0,sticky="w",padx=4,pady=2)
        Scale(parent,from_=a,to=b,orient=HORIZONTAL,resolution=res,length=360,variable=var)\
            .grid(row=r,column=1,padx=6)
