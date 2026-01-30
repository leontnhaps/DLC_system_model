#!/usr/bin/env python3
"""
UI components - Complete layout matching Com_test
"""

from tkinter import Tk, Label, Button, Frame, BooleanVar, Checkbutton, ttk, StringVar, IntVar, DoubleVar, Scale, HORIZONTAL
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

class ScanTab:
    """스캔 탭 UI"""
    def __init__(self, parent, callbacks):
        self.callbacks = callbacks
        self.frame = parent
        self._build()
    
    def _build(self):
        # 변수들
        self.pan_min = IntVar(value=-30)
        self.pan_max = IntVar(value=30)
        self.pan_step = IntVar(value=10)
        self.tilt_min = IntVar(value=0)
        self.tilt_max = IntVar(value=30)
        self.tilt_step = IntVar(value=10)
        self.width = IntVar(value=2592)
        self.height = IntVar(value=1944)
        self.quality = IntVar(value=90)
        self.speed = IntVar(value=100)
        self.acc = DoubleVar(value=1.0)
        self.settle = DoubleVar(value=0.25)
        self.led_settle = DoubleVar(value=0.15)
        
        r = 0
        self._row(r, "Pan min/max/step", self.pan_min, self.pan_max, self.pan_step); r += 1
        self._row(r, "Tilt min/max/step", self.tilt_min, self.tilt_max, self.tilt_step); r += 1
        self._row(r, "Resolution (w×h)", self.width, self.height, None, ("W","H","")); r += 1
        self._entry(r, "Quality(%)", self.quality); r += 1
        self._entry(r, "Speed", self.speed); r += 1
        self._entry(r, "Accel", self.acc); r += 1
        self._entry(r, "Settle(s)", self.settle); r += 1
        self._entry(r, "LED Settle(s)", self.led_settle); r += 1
        
        ops = Frame(self.frame)
        ops.grid(row=r, column=0, columnspan=4, sticky="w", pady=6)
        Button(ops, text="Start Scan", command=self._on_start).pack(side="left", padx=4)
        Button(ops, text="Stop Scan", command=self._on_stop).pack(side="left", padx=4)
        self.prog = ttk.Progressbar(ops, orient=HORIZONTAL, length=280, mode="determinate")
        self.prog.pack(side="left", padx=10)
        self.prog_lbl = Label(ops, text="0 / 0")
        self.prog_lbl.pack(side="left")
    
    def _row(self, r, txt, v1, v2, v3, labels=("Min","Max","Step")):
        Label(self.frame, text=txt).grid(row=r, column=0, sticky="w", padx=(5,10))
        ttk.Entry(self.frame, textvariable=v1, width=8).grid(row=r, column=1, sticky="w", padx=2)
        Label(self.frame, text=labels[0], font=("", 8)).grid(row=r, column=1, sticky="e", padx=2)
        ttk.Entry(self.frame, textvariable=v2, width=8).grid(row=r, column=2, sticky="w", padx=2)
        Label(self.frame, text=labels[1], font=("", 8)).grid(row=r, column=2, sticky="e", padx=2)
        if v3 is not None:
            ttk.Entry(self.frame, textvariable=v3, width=8).grid(row=r, column=3, sticky="w", padx=2)
            Label(self.frame, text=labels[2], font=("", 8)).grid(row=r, column=3, sticky="e", padx=2)
    
    def _entry(self, r, txt, var):
        Label(self.frame, text=txt).grid(row=r, column=0, sticky="w", padx=(5,10))
        ttk.Entry(self.frame, textvariable=var, width=12).grid(row=r, column=1, sticky="w", padx=2)
    
    def _on_start(self):
        if self.callbacks.get('start_scan'):
            params = {
                'pan_min': self.pan_min.get(),
                'pan_max': self.pan_max.get(),
                'pan_step': self.pan_step.get(),
                'tilt_min': self.tilt_min.get(),
                'tilt_max': self.tilt_max.get(),
                'tilt_step': self.tilt_step.get(),
                'width': self.width.get(),
                'height': self.height.get(),
                'quality': self.quality.get(),
                'speed': self.speed.get(),
                'acc': self.acc.get(),
                'settle': self.settle.get(),
                'led_settle': self.led_settle.get()
            }
            self.callbacks['start_scan'](params)
    
    def _on_stop(self):
        if self.callbacks.get('stop_scan'):
            self.callbacks['stop_scan']()

class ManualTab:
    """수동 제어 탭 UI"""
    def __init__(self, parent, callbacks):
        self.callbacks = callbacks
        self.frame = parent
        self._build()
    
    def _build(self):
        # 변수들
        self.mv_pan = DoubleVar(value=0.0)
        self.mv_tilt = DoubleVar(value=0.0)
        self.mv_speed = IntVar(value=100)
        self.mv_acc = DoubleVar(value=1.0)
        self.led = IntVar(value=0)
        
        self._slider(0, "Pan", -180, 180, self.mv_pan, 0.5)
        ttk.Entry(self.frame, textvariable=self.mv_pan, width=8).grid(row=0, column=2, sticky="w", padx=5)
        
        self._slider(1, "Tilt", -30, 90, self.mv_tilt, 0.5)
        ttk.Entry(self.frame, textvariable=self.mv_tilt, width=8).grid(row=1, column=2, sticky="w", padx=5)
        
        self._slider(2, "Speed", 0, 100, self.mv_speed, 1)
        self._slider(3, "Accel", 0, 1, self.mv_acc, 0.1)
        
        Button(self.frame, text="Center (0,0)", command=self._on_center).grid(row=4, column=0, sticky="w", pady=4)
        Button(self.frame, text="Apply Move", command=self._on_move).grid(row=4, column=1, sticky="e", pady=4)
        
        self._slider(5, "LED", 0, 255, self.led, 1)
        Button(self.frame, text="Set LED", command=self._on_led).grid(row=6, column=1, sticky="e", pady=4)
        Button(self.frame, text="Laser ON/OFF", command=self._on_laser).grid(row=6, column=2, sticky="w", padx=4, pady=4)
    
    def _slider(self, r, txt, mi, ma, var, res):
        Label(self.frame, text=txt).grid(row=r, column=0, sticky="w", padx=5)
        Scale(self.frame, from_=mi, to=ma, orient=HORIZONTAL, resolution=res, length=250,
              variable=var).grid(row=r, column=1, sticky="w", padx=5)
    
    def _on_center(self):
        self.mv_pan.set(0.0)
        self.mv_tilt.set(0.0)
        if self.callbacks.get('apply_move'):
            self.callbacks['apply_move'](0.0, 0.0, self.mv_speed.get(), self.mv_acc.get())
    
    def _on_move(self):
        if self.callbacks.get('apply_move'):
            self.callbacks['apply_move'](
                self.mv_pan.get(), self.mv_tilt.get(),
                self.mv_speed.get(), self.mv_acc.get()
            )
    
    def _on_led(self):
        if self.callbacks.get('set_led'):
            self.callbacks['set_led'](self.led.get())
    
    def _on_laser(self):
        if self.callbacks.get('toggle_laser'):
            self.callbacks['toggle_laser']()

class PreviewTab:
    """프리뷰 및 설정 탭 UI"""
    def __init__(self, parent, callbacks):
        self.callbacks = callbacks
        self.frame = parent
        self._build()
    
    def _build(self):
        # 변수들
        self.preview_enable = BooleanVar(value=False)
        self.preview_w = IntVar(value=640)
        self.preview_h = IntVar(value=480)
        self.preview_fps = IntVar(value=5)
        self.preview_q = IntVar(value=70)
        
        row = 0
        Checkbutton(self.frame, text="Live Preview", variable=self.preview_enable,
                   command=self._on_toggle).grid(row=row, column=0, sticky="w", pady=2); row += 1
        
        self._row(row, "Preview w/h/-", self.preview_w, self.preview_h, None, ("W","H","")); row += 1
        self._entry(row, "Preview fps", self.preview_fps); row += 1
        self._entry(row, "Preview quality", self.preview_q); row += 1
        Button(self.frame, text="Apply Preview Size",
               command=self._on_apply_size).grid(row=row, column=1, sticky="w", pady=4); row += 1
        
        ttk.Separator(self.frame, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=(8,6)); row += 1
        
        # Snap Capture
        snap_frame = Frame(self.frame)
        snap_frame.grid(row=row, column=0, columnspan=4, sticky="w", pady=6); row += 1
        Label(snap_frame, text="📸 Snap Capture:").pack(side="left", padx=3)
        Button(snap_frame, text="Capture 5MP+3MP", command=self._on_snap,
               bg="#4CAF50", fg="white", font=("", 10, "bold")).pack(side="left", padx=5)
        
        ttk.Separator(self.frame, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=(8,6)); row += 1
        
        # IR-CUT Controls
        ir_frame = Frame(self.frame)
        ir_frame.grid(row=row, column=0, columnspan=4, sticky="w", pady=6); row += 1
        Label(ir_frame, text="IR-CUT:").pack(side="left", padx=3)
        Button(ir_frame, text="☀️ Day Mode", command=lambda: self._on_ir_cut("day"),
               bg="#FFE0B2").pack(side="left", padx=2)
        Button(ir_frame, text="🌙 Night Mode", command=lambda: self._on_ir_cut("night"),
               bg="#444", fg="white").pack(side="left", padx=2)
        
        for c in range(4): self.frame.grid_columnconfigure(c, weight=1)
    
    def _row(self, r, txt, v1, v2, v3, labels=("Min","Max","Step")):
        Label(self.frame, text=txt).grid(row=r, column=0, sticky="w", padx=(5,10))
        ttk.Entry(self.frame, textvariable=v1, width=8).grid(row=r, column=1, sticky="w", padx=2)
        Label(self.frame, text=labels[0], font=("", 8)).grid(row=r, column=1, sticky="e", padx=2)
        ttk.Entry(self.frame, textvariable=v2, width=8).grid(row=r, column=2, sticky="w", padx=2)
        Label(self.frame, text=labels[1], font=("", 8)).grid(row=r, column=2, sticky="e", padx=2)
        if v3 is not None:
            ttk.Entry(self.frame, textvariable=v3, width=8).grid(row=r, column=3, sticky="w", padx=2)
            Label(self.frame, text=labels[2], font=("", 8)).grid(row=r, column=3, sticky="e", padx=2)
    
    def _entry(self, r, txt, var):
        Label(self.frame, text=txt).grid(row=r, column=0, sticky="w", padx=(5,10))
        ttk.Entry(self.frame, textvariable=var, width=12).grid(row=r, column=1, sticky="w", padx=2)
    
    def _on_toggle(self):
        if self.callbacks.get('toggle_preview'):
            self.callbacks['toggle_preview'](self.preview_enable.get(),
                self.preview_w.get(), self.preview_h.get(),
                self.preview_fps.get(), self.preview_q.get())
    
    def _on_apply_size(self):
        if self.preview_enable.get():
            # 재시작
            self._on_toggle()
    
    def _on_snap(self):
        if self.callbacks.get('snap_capture'):
            self.callbacks['snap_capture']()
    
    def _on_ir_cut(self, mode):
        if self.callbacks.get('set_ir_cut'):
            self.callbacks['set_ir_cut'](mode)
