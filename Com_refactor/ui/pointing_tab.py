"""Pointing tab UI component."""

from tkinter import Label, Button, Frame, StringVar, Canvas, Scrollbar
from tkinter import ttk


class PointingTab:
    """Pointing 탭 UI (스크롤 가능)"""
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
        r += 1
        
        # ⭐ Aiming 상태 표시
        self.aim_status_label = Label(self.frame, text="", font=("", 10), fg="#333")
        self.aim_status_label.grid(row=r, column=0, columnspan=3, sticky="w", padx=10, pady=5)
        r += 1

        # Aiming 모드 (고정: adaptive)
        Label(self.frame, text="Aiming Mode:").grid(row=r, column=0, sticky="w", padx=(5, 10), pady=3)
        Label(self.frame, text="adaptive (fixed)", fg="#666").grid(row=r, column=1, sticky="w", padx=2)
        if self.callbacks.get('set_pointing_mode'):
            self.callbacks['set_pointing_mode']("adaptive")
        r += 1

        # 선택된 타깃에 대해 Start로 정밀 조준 시작
        self._selected_track_id = None
        self.btn_start_aim = Button(
            self.frame, text="▶ Start Aiming",
            command=self._on_start_aiming,
            width=15, bg="#4CAF50", fg="white",
            font=("", 10, "bold"), state="disabled"
        )
        self.btn_start_aim.grid(row=r, column=0, columnspan=3, pady=5)
        r += 1
        
        # Stop Aiming 버튼
        self.btn_stop_aim = Button(self.frame, text="⛔ Stop Aiming", 
                                    command=self._on_stop_aiming,
                                    width=15, bg="#F44336", fg="white", 
                                    font=("", 10, "bold"), state="disabled")
        self.btn_stop_aim.grid(row=r, column=0, columnspan=3, pady=5)
        r += 1
        
        # ⭐ Debug Preview (400x400)
        Label(self.frame, text="📹 Detection Debug", font=("", 10, "bold")).grid(row=r, column=0, columnspan=3, pady=(10,2))
        r += 1
        debug_frame = Frame(self.frame, width=400, height=400, bg="#111",
                            highlightthickness=1, highlightbackground="#333")
        debug_frame.grid(row=r, column=0, columnspan=3, pady=5)
        debug_frame.pack_propagate(False)
        self.debug_preview_label = Label(debug_frame, bg="#111", fg="#666",
                                          text="(Waiting for aiming...)")
        self.debug_preview_label.pack(fill="both", expand=True)
        r += 1
        
        # 오차 텍스트 Label
        self.debug_error_label = Label(self.frame, text="", font=("", 11, "bold"), fg="#888")
        self.debug_error_label.grid(row=r, column=0, columnspan=3)
        r += 1
        
        # ⭐ Laser Diff Preview (400x300)
        Label(self.frame, text="🔴 Laser Diff", font=("", 10, "bold")).grid(row=r, column=0, columnspan=3, pady=(10,2))
        r += 1
        laser_frame = Frame(self.frame, width=400, height=300, bg="#111",
                            highlightthickness=1, highlightbackground="#333")
        laser_frame.grid(row=r, column=0, columnspan=3, pady=5)
        laser_frame.pack_propagate(False)
        self.laser_diff_label = Label(laser_frame, bg="#111", fg="#666",
                                       text="(Waiting for laser diff...)")
        self.laser_diff_label.pack(fill="both", expand=True)
    
    def _on_choose_csv(self):
        if self.callbacks.get('pointing_choose_csv'):
            self.callbacks['pointing_choose_csv']()
    
    def _on_compute(self):
        if self.callbacks.get('pointing_compute'):
            self.callbacks['pointing_compute']()

    def _on_stop_aiming(self):
        if self.callbacks.get('stop_aiming'):
            self.callbacks['stop_aiming']()
        self.btn_stop_aim.config(state="disabled")
        if self._selected_track_id is not None:
            self.btn_start_aim.config(state="normal")
        self.aim_status_label.config(text="⛔ 조준 중단됨", fg="red")
    
    def _create_target_buttons(self, targets):
        """Track ID별 버튼 생성
        
        Args:
            targets: {track_id: (pan, tilt), ...}
        """
        # 기존 버튼 제거
        for widget in self.buttons_frame.winfo_children():
            widget.destroy()
        self.target_value_labels = {}

        self._selected_track_id = None
        if hasattr(self, 'btn_start_aim'):
            self.btn_start_aim.config(state="disabled")
        if hasattr(self, 'btn_stop_aim'):
            self.btn_stop_aim.config(state="disabled")
        
        if not targets:
            Label(self.buttons_frame, text="No targets computed", fg="gray").pack()
            return
        
        # Track ID별 버튼 생성
        for track_id in sorted(targets.keys()):
            pan, tilt = targets[track_id]
            row_frame = Frame(self.buttons_frame)
            row_frame.pack(anchor="w", fill="x", pady=2)

            btn_text = f"🎯 ID {track_id}"
            btn = Button(
                row_frame,
                text=btn_text,
                command=lambda tid=track_id: self._on_aim_target(tid),
                width=12,
                bg="#2196F3",
                fg="white",
                font=("", 9, "bold")
            )
            btn.pack(side="left", padx=(0, 8))
            
            # Tooltip (Pan/Tilt 표시)
            label = Label(row_frame, text=f"({pan}°, {tilt}°)", font=("", 8), fg="gray")
            label.pack(side="left")
            self.target_value_labels[track_id] = label

    def update_target_value(self, track_id, pan, tilt):
        """특정 Track ID의 좌표 라벨만 갱신"""
        if not hasattr(self, "target_value_labels"):
            return
        label = self.target_value_labels.get(track_id)
        if label is None:
            return
        label.config(text=f"({pan}°, {tilt}°)")
    
    def _on_aim_target(self, track_id):
        """ID 버튼 클릭 → 초기 위치 이동만 수행"""
        if self.callbacks.get('move_to_target'):
            self._selected_track_id = track_id
            self.btn_start_aim.config(state="normal")
            self.btn_stop_aim.config(state="disabled")
            self.aim_status_label.config(
                text=f"🎯 Track {track_id} 초기 위치 이동. Start Aiming을 누르세요.",
                fg="blue"
            )
            self.callbacks['move_to_target'](track_id)

    def _on_start_aiming(self):
        """선택된 타깃으로 정밀 조준 시작"""
        if self._selected_track_id is None:
            self.aim_status_label.config(text="⚠️ 먼저 Target(ID) 버튼을 선택하세요.", fg="orange")
            return

        if self.callbacks.get('start_aiming'):
            started = self.callbacks['start_aiming'](self._selected_track_id)
            if started:
                self.btn_start_aim.config(state="disabled")
                self.btn_stop_aim.config(state="normal")
                self.aim_status_label.config(
                    text=f"🎯 Track {self._selected_track_id} 조준 시작...",
                    fg="blue"
                )
    
    def _on_move_to_target(self, track_id):
        if self.callbacks.get('move_to_target'):
            self.callbacks['move_to_target'](track_id)
    
    def update_aim_status(self, track_id, iteration, message):
        """조준 상태 업데이트 (pointing_handler에서 호출)"""
        self.aim_status_label.config(text=f"🎯 [{track_id}] {message}")
        if "수렴 완료" in message or "❌" in message:
            self.btn_stop_aim.config(state="disabled")
            if self._selected_track_id is not None:
                self.btn_start_aim.config(state="normal")
            if "수렴 완료" in message:
                self.aim_status_label.config(fg="green")
            else:
                self.aim_status_label.config(fg="red")
