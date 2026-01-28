#!/usr/bin/env python3
"""
UI components and layout
"""

from tkinter import Tk, Label, Button, Frame, BooleanVar, Checkbutton
from PIL import Image, ImageTk
import io

class PreviewFrame:
    """프리뷰 디스플레이 프레임"""
    def __init__(self, parent, width=640, height=480):
        self.width = width
        self.height = height
        
        self.frame = Frame(parent, width=width, height=height,
                          bg="#111", highlightthickness=1, highlightbackground="#333")
        self.frame.pack()
        self.frame.pack_propagate(False)
        
        self.label = Label(self.frame, bg="#111")
        self.label.place(x=0, y=0, width=width, height=height)
    
    def display_image(self, jpeg_bytes):
        """이미지 표시"""
        try:
            img = Image.open(io.BytesIO(jpeg_bytes))
            img.thumbnail((self.width, self.height), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            self.label.config(image=tk_img)
            self.label.image = tk_img
        except Exception as e:
            print(f"[DISPLAY] 오류: {e}")

class ControlPanel:
    """제어 패널 (프리뷰, IR-CUT 등)"""
    def __init__(self, parent, callbacks):
        """
        callbacks: {
            'toggle_preview': callback,
            'start_preview': callback,
            'set_ir_cut': callback
        }
        """
        self.callbacks = callbacks
        
        self.frame = Frame(parent)
        self.frame.pack(fill="x", padx=10, pady=10)
        
        # 프리뷰 컨트롤
        self.preview_enable = BooleanVar(value=False)
        Checkbutton(self.frame, text="프리뷰 활성화", variable=self.preview_enable,
                   command=self._on_preview_toggle).pack(side="left", padx=5)
        
        Button(self.frame, text="프리뷰 시작", 
               command=self._on_preview_start).pack(side="left", padx=5)
        
        # IR-CUT 컨트롤
        ir_frame = Frame(self.frame, relief="ridge", borderwidth=1)
        ir_frame.pack(side="left", padx=10)
        Label(ir_frame, text="IR-CUT:").pack(side="left", padx=3)
        Button(ir_frame, text="☀️ Day Mode", 
               command=lambda: self._on_ir_cut("day"),
               bg="#FFE0B2").pack(side="left", padx=2)
        Button(ir_frame, text="🌙 Night Mode", 
               command=lambda: self._on_ir_cut("night"),
               bg="#444", fg="white").pack(side="left", padx=2)
    
    def _on_preview_toggle(self):
        if self.callbacks.get('toggle_preview'):
            self.callbacks['toggle_preview'](self.preview_enable.get())
    
    def _on_preview_start(self):
        if self.callbacks.get('start_preview'):
            self.callbacks['start_preview']()
    
    def _on_ir_cut(self, mode):
        if self.callbacks.get('set_ir_cut'):
            self.callbacks['set_ir_cut'](mode)
