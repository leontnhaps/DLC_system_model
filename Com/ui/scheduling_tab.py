"""Scheduling tab UI component."""

from tkinter import Label, Button, DoubleVar
from tkinter import ttk


class SchedulingTab:
    """Scheduling 탭 UI"""
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

        self.dwell_seconds = DoubleVar(value=20.0)
        Label(self.frame, text="Shoot Timer (s)").grid(row=r, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(self.frame, textvariable=self.dwell_seconds, width=10).grid(row=r, column=1, sticky="w", padx=8, pady=4)
        r += 1
        
        self.led_probe_seconds = DoubleVar(value=10.0)
        Label(self.frame, text="Battery Check (s)").grid(row=r, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(self.frame, textvariable=self.led_probe_seconds, width=10).grid(row=r, column=1, sticky="w", padx=8, pady=4)
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
        self.btn_stop.grid(row=r, column=1, sticky="w", padx=8, pady=4)
        r += 1

        self.status_label = Label(self.frame, text="대기 중", fg="#333", font=("", 10))
        self.status_label.grid(row=r, column=0, columnspan=2, sticky="w", padx=8, pady=(10, 6))

        for c in range(2):
            self.frame.grid_columnconfigure(c, weight=1)

    def _on_roundrobin(self):
        if self.callbacks.get("start_roundrobin"):
            self.callbacks["start_roundrobin"]()

    def _on_stop(self):
        if self.callbacks.get("stop_scheduling"):
            self.callbacks["stop_scheduling"]()

    def set_running_state(self, is_running):
        if is_running:
            self.btn_roundrobin.config(state="disabled")
            self.btn_stop.config(state="normal")
        else:
            self.btn_roundrobin.config(state="normal")
            self.btn_stop.config(state="disabled")

    def update_status(self, text, fg="#333"):
        self.status_label.config(text=text, fg=fg)

    def get_dwell_seconds(self):
        try:
            return max(0.2, float(self.dwell_seconds.get()))
        except Exception:
            return 20.0
    
    def get_led_probe_seconds(self):
        try:
            return max(0.5, float(self.led_probe_seconds.get()))
        except Exception:
            return 10.0
