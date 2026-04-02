"""Scheduling tab UI component."""

from tkinter import Label, Button, DoubleVar, BooleanVar
from tkinter import ttk


class SchedulingTab:
    """Scheduling 탭 UI"""
    DEFAULT_PROGRESS_TEXT = (
        "모드: -\n"
        "전체 경과: -\n"
        "진행: -\n"
        "현재 Frame: -\n"
        "현재 타깃: -\n"
        "현재 작업: -"
    )

    def __init__(self, parent, callbacks):
        self.callbacks = callbacks
        self.frame = parent
        self._build()

    def _build(self):
        r = 0
        Label(self.frame, text="Scheduling", font=("", 12, "bold")).grid(
            row=r, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 12)
        )
        r += 1

        self.frame_seconds = DoubleVar(value=0.0)
        Label(self.frame, text="T_frame_sec (s)").grid(row=r, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(self.frame, textvariable=self.frame_seconds, width=10).grid(row=r, column=1, sticky="w", padx=8, pady=4)
        r += 1

        self.total_seconds = DoubleVar(value=0.0)
        Label(self.frame, text="T_total_sec (s)").grid(row=r, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(self.frame, textvariable=self.total_seconds, width=10).grid(row=r, column=1, sticky="w", padx=8, pady=4)
        r += 1

        self.led_probe_seconds = DoubleVar(value=10.0)
        Label(self.frame, text="Battery Check (s)").grid(row=r, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(self.frame, textvariable=self.led_probe_seconds, width=10).grid(row=r, column=1, sticky="w", padx=8, pady=4)
        r += 1

        self.enable_battery_check = BooleanVar(value=False)
        ttk.Checkbutton(
            self.frame,
            text="배터리 상태 체크 사용",
            variable=self.enable_battery_check,
        ).grid(row=r, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))
        r += 1

        self.btn_roundrobin = Button(
            self.frame,
            text="RoundRobin",
            command=self._on_roundrobin,
            width=18,
            bg="#1976D2",
            fg="white",
            font=("", 10, "bold"),
        )
        self.btn_roundrobin.grid(row=r, column=0, sticky="w", padx=8, pady=4)

        self.btn_proposed = Button(
            self.frame,
            text="Proposed",
            command=self._on_proposed,
            width=18,
            bg="#388E3C",
            fg="white",
            font=("", 10, "bold"),
        )
        self.btn_proposed.grid(row=r, column=1, sticky="w", padx=8, pady=4)
        r += 1

        self.btn_stop = Button(
            self.frame,
            text="Stop Scheduling",
            command=self._on_stop,
            width=18,
            bg="#D32F2F",
            fg="white",
            font=("", 10, "bold"),
            state="disabled",
        )
        self.btn_stop.grid(row=r, column=0, columnspan=2, sticky="w", padx=8, pady=4)
        r += 1

        self.status_label = Label(self.frame, text="대기 중", fg="#333", font=("", 10))
        self.status_label.grid(row=r, column=0, columnspan=2, sticky="w", padx=8, pady=(10, 6))
        r += 1

        self.progress_label = Label(
            self.frame,
            text=self.DEFAULT_PROGRESS_TEXT,
            fg="#555",
            font=("", 9),
            justify="left",
            anchor="w",
        )
        self.progress_label.grid(row=r, column=0, columnspan=2, sticky="we", padx=8, pady=(0, 6))

        for c in range(2):
            self.frame.grid_columnconfigure(c, weight=1)

    def _on_roundrobin(self):
        if self.callbacks.get("start_roundrobin"):
            self.callbacks["start_roundrobin"]()

    def _on_proposed(self):
        if self.callbacks.get("start_proposed"):
            self.callbacks["start_proposed"]()

    def _on_stop(self):
        if self.callbacks.get("stop_scheduling"):
            self.callbacks["stop_scheduling"]()

    def set_running_state(self, is_running):
        if is_running:
            self.btn_roundrobin.config(state="disabled")
            self.btn_proposed.config(state="disabled")
            self.btn_stop.config(state="normal")
        else:
            self.btn_roundrobin.config(state="normal")
            self.btn_proposed.config(state="normal")
            self.btn_stop.config(state="disabled")

    def update_status(self, text, fg="#333"):
        self.status_label.config(text=text, fg=fg)

    def update_progress_text(self, text, fg="#555"):
        self.progress_label.config(text=text, fg=fg)

    def reset_progress_text(self):
        self.update_progress_text(self.DEFAULT_PROGRESS_TEXT)

    def get_frame_seconds(self):
        try:
            value = float(self.frame_seconds.get())
        except Exception:
            return 0.0
        return value if value > 0.0 else 0.0

    def get_total_seconds(self):
        try:
            value = float(self.total_seconds.get())
        except Exception:
            return 0.0
        return value if value > 0.0 else 0.0

    def get_led_probe_seconds(self):
        try:
            return max(0.5, float(self.led_probe_seconds.get()))
        except Exception:
            return 10.0

    def get_battery_check_enabled(self):
        try:
            return bool(self.enable_battery_check.get())
        except Exception:
            return False
