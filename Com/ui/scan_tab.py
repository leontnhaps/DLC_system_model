"""Scan tab UI component."""

from tkinter import Label, Button, Frame, StringVar, IntVar, DoubleVar, HORIZONTAL
from tkinter import ttk


class ScanTab:
    """스캔 탭 UI"""
    def __init__(self, parent, callbacks):
        self.callbacks = callbacks
        self.frame = parent
        self._build()
    
    def _build(self):
        # 변수들
        self.pan_min = IntVar(value=-30)
        self.pan_max = IntVar(value=0)
        self.pan_step = IntVar(value=5)
        self.tilt_min = IntVar(value=-10)
        self.tilt_max = IntVar(value=5)
        self.tilt_step = IntVar(value=5)
        self.scan_resolution = StringVar(value="5MP (2592×1944)")
        self.width = IntVar(value=2592)
        self.height = IntVar(value=1944)
        self.quality = IntVar(value=90)
        self.speed = IntVar(value=0)
        self.acc = DoubleVar(value=0.0)
        self.settle = DoubleVar(value=0.25)
        self.led_settle = DoubleVar(value=0.2)

        
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
            self.callbacks['start_scan'](self.get_scan_params())

    def get_scan_params(self):
        """현재 Scan UI 값을 dict로 반환"""
        return {
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
            'yolo_weights': self.yolo_weights.get()
        }
    
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
