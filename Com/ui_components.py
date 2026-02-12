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
        self.scan_resolution = StringVar(value="5MP (2592×1944)")
        self.width = IntVar(value=2592)
        self.height = IntVar(value=1944)
        self.quality = IntVar(value=90)
        self.speed = IntVar(value=0)
        self.acc = DoubleVar(value=0.0)
        self.settle = DoubleVar(value=0.4)
        self.led_settle = DoubleVar(value=0.4)

        
        r = 0
        self._row(r, "Pan min/max/step", self.pan_min, self.pan_max, self.pan_step); r += 1
        self._row(r, "Tilt min/max/step", self.tilt_min, self.tilt_max, self.tilt_step); r += 1
        
        # Resolution Combobox
        Label(self.frame, text="Resolution").grid(row=r, column=0, sticky="w", padx=(5,10))
        res_combo = ttk.Combobox(self.frame, textvariable=self.scan_resolution, state="readonly", width=20)
        res_combo['values'] = (
            "VGA (640×480)",
            "1.3MP (1296×972)",
            "Full HD (1920×1080)",
            "5MP (2592×1944)"
        )
        res_combo.grid(row=r, column=1, columnspan=2, sticky="w", padx=2)
        res_combo.bind("<<ComboboxSelected>>", self._on_scan_resolution_change); r += 1
        
        self._entry(r, "Quality(%)", self.quality); r += 1
        self._entry(r, "Speed", self.speed); r += 1
        self._entry(r, "Accel", self.acc); r += 1
        self._entry(r, "Settle(s)", self.settle); r += 1
        self._entry(r, "LED Settle(s)", self.led_settle); r += 1
        
        # YOLO weights 경로
        self.yolo_weights = StringVar(value="../yolov11m_diff.pt")
        self._entry(r, "YOLO Weights", self.yolo_weights); r += 1
        
        ops = Frame(self.frame)
        ops.grid(row=r, column=0, columnspan=4, sticky="w", pady=6)
        self.btn_start = Button(ops, text="Start Scan", command=self._on_start)
        self.btn_start.pack(side="left", padx=4)
        self.btn_stop = Button(ops, text="Stop Scan", command=self._on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=4)
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
                'led_settle': self.led_settle.get(),
                'yolo_weights': self.yolo_weights.get()  # YOLO weights 추가
            }
            self.callbacks['start_scan'](params)
    
    def _on_stop(self):
        if self.callbacks.get('stop_scan'):
            self.callbacks['stop_scan']()
            
    def set_scan_state(self, is_scanning):
        """스캔 중 버튼 상태 변경"""
        if is_scanning:
            self.btn_start.config(state="disabled")
            self.btn_stop.config(state="normal")
        else:
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
    
    def _on_scan_resolution_change(self, event=None):
        """Resolution combobox 변경 시 width/height 자동 설정"""
        res_map = {
            "VGA (640×480)": (640, 480),
            "1.3MP (1296×972)": (1296, 972),
            "Full HD (1920×1080)": (1920, 1080),
            "5MP (2592×1944)": (2592, 1944)
        }
        selected = self.scan_resolution.get()
        if selected in res_map:
            w, h = res_map[selected]
            self.width.set(w)
            self.height.set(h)

class TestSettingsTab:
    """Test & Settings 탭 - Manual + Preview 통합"""
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
        self.preview_enable = BooleanVar(value=False)
        self.preview_resolution = StringVar(value="VGA (640×480)")
        self.preview_w = IntVar(value=640)
        self.preview_h = IntVar(value=480)
        self.preview_fps = IntVar(value=5)
        self.preview_q = IntVar(value=70)
        
        row = 0
        
        # ========== Pan/Tilt Control ==========
        Label(self.frame, text="🎮 Pan/Tilt Control", font=("", 11, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", padx=5, pady=(5, 10)); row += 1
        
        self._slider(row, "Pan", -180, 180, self.mv_pan, 0.5); row += 1
        ttk.Entry(self.frame, textvariable=self.mv_pan, width=8).grid(row=row-1, column=2, sticky="w", padx=5)
        
        self._slider(row, "Tilt", -30, 90, self.mv_tilt, 0.5); row += 1
        ttk.Entry(self.frame, textvariable=self.mv_tilt, width=8).grid(row=row-1, column=2, sticky="w", padx=5)
        
        self._slider(row, "Speed", 0, 100, self.mv_speed, 1); row += 1
        self._slider(row, "Accel", 0, 1, self.mv_acc, 0.1); row += 1
        
        btn_frame = Frame(self.frame)
        btn_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=4); row += 1
        Button(btn_frame, text="Center (0,0)", command=self._on_center).pack(side="left", padx=5)
        Button(btn_frame, text="Apply Move", command=self._on_move, bg="#2196F3", fg="white").pack(side="left", padx=5)
        
        ttk.Separator(self.frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10); row += 1
        
        # ========== LED & Laser ==========
        Label(self.frame, text="💡 LED & Laser", font=("", 11, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", padx=5, pady=(5, 10)); row += 1
        
        self._slider(row, "LED", 0, 255, self.led, 1); row += 1
        
        led_btn_frame = Frame(self.frame)
        led_btn_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=4); row += 1
        Button(led_btn_frame, text="Set LED", command=self._on_led).pack(side="left", padx=5)
        Button(led_btn_frame, text="Laser ON/OFF", command=self._on_laser, bg="#FF5722", fg="white").pack(side="left", padx=5)
        
        ttk.Separator(self.frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10); row += 1
        
        # ========== Preview Settings ==========
        Label(self.frame, text="📹 Preview Settings", font=("", 11, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", padx=5, pady=(5, 10)); row += 1
        
        # Live Preview 체크박스 (명시적으로 False로 강제)
        self.preview_enable.set(False)  # 강제 초기화!
        preview_check = Checkbutton(self.frame, text="Live Preview", variable=self.preview_enable,
                   command=self._on_toggle)
        preview_check.grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        preview_check.deselect()  # 체크 해제
        row += 1
        
        # Preview Resolution Combobox
        Label(self.frame, text="Resolution").grid(row=row, column=0, sticky="w", padx=(5,10))
        prev_res_combo = ttk.Combobox(self.frame, textvariable=self.preview_resolution, state="readonly", width=20)
        prev_res_combo['values'] = (
            "VGA (640×480)",
            "1.3MP (1296×972)",
            "Full HD (1920×1080)",
            "5MP (2592×1944)"
        )
        prev_res_combo.grid(row=row, column=1, columnspan=2, sticky="w", padx=2)
        prev_res_combo.bind("<<ComboboxSelected>>", self._on_preview_resolution_change); row += 1
        
        self._entry(row, "Preview fps", self.preview_fps); row += 1
        self._entry(row, "Preview quality", self.preview_q); row += 1
        
        ttk.Separator(self.frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10); row += 1
        
        # ========== Capture ==========
        Label(self.frame, text="📸 Capture", font=("", 11, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", padx=5, pady=(5, 10)); row += 1
        
        Button(self.frame, text="Snap (Preview Resolution)", command=self._on_snap,
               bg="#4CAF50", fg="white", font=("", 10, "bold"), width=25).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=5, pady=5); row += 1
        
        ttk.Separator(self.frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10); row += 1
        
        # ========== IR-CUT Control ==========
        Label(self.frame, text="🌓 IR-CUT Control", font=("", 11, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", padx=5, pady=(5, 10)); row += 1
        
        ir_frame = Frame(self.frame)
        ir_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=5); row += 1
        Button(ir_frame, text="🔍 Normal", command=lambda: self._on_ir_cut("night"),
               width=10, bg="#9E9E9E", fg="white", font=("", 10, "bold")).pack(side="left", padx=5)
        Button(ir_frame, text="🔴 IR Mode", command=lambda: self._on_ir_cut("day"),
               width=10, bg="#E57373", fg="white", font=("", 10, "bold")).pack(side="left", padx=5)
        
        for c in range(3): self.frame.grid_columnconfigure(c, weight=1)
    
    def _slider(self, r, txt, mi, ma, var, res):
        Label(self.frame, text=txt).grid(row=r, column=0, sticky="w", padx=5)
        Scale(self.frame, from_=mi, to=ma, orient=HORIZONTAL, resolution=res, length=250,
              variable=var).grid(row=r, column=1, sticky="w", padx=5)
    
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
    
    # Pan/Tilt callbacks
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
    
    # Preview callbacks
    def _on_toggle(self):
        if self.callbacks.get('toggle_preview'):
            self.callbacks['toggle_preview'](self.preview_enable.get(),
                self.preview_w.get(), self.preview_h.get(),
                self.preview_fps.get(), self.preview_q.get())
    
    def _on_apply_size(self):
        if self.preview_enable.get():
            self._on_toggle()
    
    def _on_ir_cut(self, mode):
        if self.callbacks.get('set_ir_cut'):
            self.callbacks['set_ir_cut'](mode)
    
    def _on_snap(self):
        if self.callbacks.get('snap_capture'):
            self.callbacks['snap_capture']()

    def _on_preview_resolution_change(self, event=None):
        """Preview resolution combobox 변경 시 자동 적용 (Auto)"""
        res_map = {
            "VGA (640×480)": (640, 480),
            "1.3MP (1296×972)": (1296, 972),
            "Full HD (1920×1080)": (1920, 1080),
            "5MP (2592×1944)": (2592, 1944)
        }
        selected = self.preview_resolution.get()
        if selected in res_map:
            w, h = res_map[selected]
            self.preview_w.set(w)
            self.preview_h.set(h)
            # 프리뷰 중이면 자동 재시작
            if self.preview_enable.get():
                self._on_toggle()
    
    def update_info(self, text):
        """정보 라벨 업데이트"""
        if hasattr(self, 'info_label'):
            self.info_label.config(text=text)


class PointingTab:
    """Pointing 탭 UI"""
    def __init__(self, parent, callbacks):
        self.callbacks = callbacks
        self.frame = parent
        self._build()
    
    def _build(self):
        # 변수들
        self.point_csv_path = StringVar(value="")
        
        # CSV 선택
        r = 0
        Label(self.frame, text="CSV 파일:").grid(row=r, column=0, sticky="w", padx=(5,10), pady=5)
        ttk.Entry(self.frame, textvariable=self.point_csv_path, width=40).grid(row=r, column=1, sticky="w", padx=2)
        Button(self.frame, text="Browse", command=self._on_choose_csv, width=10).grid(row=r, column=2, sticky="w", padx=5)
        r += 1
        
        # Compute 버튼
        Button(self.frame, text="🔍 Compute Targets", command=self._on_compute, 
               width=20, bg="#4CAF50", fg="white", font=("", 10, "bold")).grid(row=r, column=0, columnspan=3, pady=10)
        r += 1
        
        # Track ID 버튼 영역
        Label(self.frame, text="Targets:").grid(row=r, column=0, sticky="w", padx=(5,10), pady=5)
        r += 1
        
        self.buttons_frame = Frame(self.frame)
        self.buttons_frame.grid(row=r, column=0, columnspan=3, sticky="w", padx=10, pady=5)
    
    def _on_choose_csv(self):
        if self.callbacks.get('pointing_choose_csv'):
            self.callbacks['pointing_choose_csv']()
    
    def _on_compute(self):
        if self.callbacks.get('pointing_compute'):
            self.callbacks['pointing_compute']()
    
    def _create_target_buttons(self, targets):
        """Track ID별 버튼 생성
        
        Args:
            targets: {track_id: (pan, tilt), ...}
        """
        # 기존 버튼 제거
        for widget in self.buttons_frame.winfo_children():
            widget.destroy()
        
        if not targets:
            Label(self.buttons_frame, text="No targets computed", fg="gray").pack()
            return
        
        # Track ID별 버튼 생성
        for track_id in sorted(targets.keys()):
            pan, tilt = targets[track_id]
            btn_text = f"ID {track_id}"
            btn = Button(
                self.buttons_frame,
                text=btn_text,
                command=lambda tid=track_id: self._on_move_to_target(tid),
                width=12,
                bg="#2196F3",
                fg="white",
                font=("", 9, "bold")
            )
            btn.pack(side="left", padx=5, pady=2)
            
            # Tooltip (Pan/Tilt 표시)
            label = Label(self.buttons_frame, text=f"({pan}°, {tilt}°)", font=("", 8), fg="gray")
            label.pack(side="left", padx=(0, 10))
    
    def _on_move_to_target(self, track_id):
        if self.callbacks.get('move_to_target'):
            self.callbacks['move_to_target'](track_id)
    
