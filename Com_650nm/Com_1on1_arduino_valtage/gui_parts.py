#!/usr/bin/env python3
"""
Reusable GUI components
"""

import tkinter as tk
from tkinter import Frame, ttk


class ScrollFrame(Frame):
    """Scrollable frame widget"""
    
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.body = Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")

        # 내용 바뀌면 스크롤영역 갱신
        self.body.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        # 부모 크기 바뀌면 내부 프레임 폭 맞춤
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfigure(self._win, width=e.width)
        )
        # 마우스 휠 스크롤
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._on_wheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _on_wheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
