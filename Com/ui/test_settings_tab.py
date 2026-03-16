"""Test settings tab UI component."""

from tkinter import Label, Button, Frame, BooleanVar, Checkbutton, StringVar, IntVar, DoubleVar, Scale, HORIZONTAL, Canvas, Scrollbar
from tkinter import ttk


class TestSettingsTab:
    """Test & Settings 탭 - Manual + Preview 통합 (스크롤 가능)"""
    def __init__(self, parent, callbacks):
        self.callbacks = callbacks
        
        # 스크롤 가능한 Canvas
        canvas = Canvas(parent)
        scrollbar = Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.frame = Frame(canvas)
        
        self.frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 마우스 휠 바인딩
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
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

        Label(self.frame, text="☀️ Exposure (Preview/Snap)", font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", padx=5, pady=(8, 6)); row += 1

        self.exposure_manual = BooleanVar(value=False)
        self.shutter_speed = IntVar(value=1000)   # µs (default 1ms)
        self.analogue_gain = DoubleVar(value=1.0) # 1.0 ~ 16.0

        chk_manual = Checkbutton(
            self.frame,
            text="Manual Exposure (Preview/Snap)",
            variable=self.exposure_manual
        )
        chk_manual.grid(row=row, column=0, columnspan=2, sticky="w", padx=5); row += 1
        
        self._entry(row, "Shutter(µs)", self.shutter_speed); row += 1
        self._slider(row, "Gain", 1.0, 16.0, self.analogue_gain, 0.1); row += 1
        Button(
            self.frame,
            text="Apply Exposure To Live Preview",
            command=self._on_apply_preview_exposure,
            width=25
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 6)); row += 1
        
        ttk.Separator(self.frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10); row += 1
        
        # ========== Capture ==========
        Label(self.frame, text="📸 Capture", font=("", 11, "bold")).grid(
            row=row, column=0, columnspan=3, sticky="w", padx=5, pady=(5, 10)); row += 1
        
        Button(self.frame, text="Snap (Preview + Exposure)", command=self._on_snap,
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

    def _on_apply_preview_exposure(self):
        if self.preview_enable.get():
            self._on_toggle()
    
    def _on_ir_cut(self, mode):
        if self.callbacks.get('set_ir_cut'):
            self.callbacks['set_ir_cut'](mode)
    
    def _on_snap(self):
        if self.callbacks.get('snap_capture'):
            # 파라미터 전달 (shutter_speed, analogue_gain)
            self.callbacks['snap_capture']()
            
    def get_exposure_params(self):
        """노출 파라미터 반환 (Preview/Snap 공용, Manual일 때만 값 반환)"""
        shutter = None
        gain = None
        if self.exposure_manual.get():
            shutter = int(self.shutter_speed.get())
            gain = float(self.analogue_gain.get())
        return shutter, gain

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
