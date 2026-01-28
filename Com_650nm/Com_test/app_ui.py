#!/usr/bin/env python3
"""
GUI layout and initialization
Separates UI setup from business logic
"""

from tkinter import Tk, Label, Button, Frame, Checkbutton, ttk, StringVar, IntVar, DoubleVar, BooleanVar, Scale, HORIZONTAL
import tkinter as tk
from gui_parts import ScrollFrame
import pathlib

# matplotlib 제거됨 (아두이노 PV Monitor)

class AppUIMixin:
    """GUI layout and widget initialization"""
    
    def setup_ui(self):
        """Initialize all UI components"""
        self._setup_scan_tab()
        self._setup_manual_tab()
        self._setup_misc_tab()
        self._setup_pointing_tab()
        # _setup_pv_tab() 제거됨 (아두이노)
    
    def _setup_scan_tab(self):
        for i in range(self.notebook.index("end")):
            if self.notebook.tab(i, "text") == "Scan":
                tab_scan = self.notebook.nametowidget(self.notebook.tabs()[i])
                break
        
        for widget in tab_scan.winfo_children(): widget.destroy()
        
        r = 0
        self._row(tab_scan, r, "Pan min/max/step", self.pan_min, self.pan_max, self.pan_step); r += 1
        self._row(tab_scan, r, "Tilt min/max/step", self.tilt_min, self.tilt_max, self.tilt_step); r += 1
        self._row(tab_scan, r, "Resolution (w×h)", self.width, self.height, None, ("W","H","")); r += 1
        self._entry(tab_scan, r, "Quality(%)", self.quality); r += 1
        self._entry(tab_scan, r, "Speed", self.speed); r += 1
        self._entry(tab_scan, r, "Accel", self.acc); r += 1
        self._entry(tab_scan, r, "Settle(s)", self.settle); r += 1
        self._entry(tab_scan, r, "LED Settle(s)", self.led_settle); r += 1
        
        ops = Frame(tab_scan); ops.grid(row=r, column=0, columnspan=4, sticky="w", pady=6)
        Button(ops, text="Start Scan", command=self.start_scan).pack(side="left", padx=4)
        Button(ops, text="Stop Scan", command=self.stop_scan).pack(side="left", padx=4)
        self.prog = ttk.Progressbar(ops, orient=HORIZONTAL, length=280, mode="determinate"); self.prog.pack(side="left", padx=10)
        self.prog_lbl = Label(ops, text="0 / 0"); self.prog_lbl.pack(side="left")
        self.last_lbl = Label(ops, text="Last: -"); self.last_lbl.pack(side="left", padx=10)
        self.dl_lbl = Label(ops, text="DL 0"); self.dl_lbl.pack(side="left", padx=10)
    
    def _setup_manual_tab(self):
        for i in range(self.notebook.index("end")):
            if self.notebook.tab(i, "text") == "Manual / LED":
                tab_manual = self.notebook.nametowidget(self.notebook.tabs()[i])
                break
        
        for widget in tab_manual.winfo_children(): widget.destroy()
        
        self._slider(tab_manual, 0, "Pan", -180, 180, self.mv_pan, 0.5)
        ttk.Entry(tab_manual, textvariable=self.mv_pan,width=8).grid(row=0,column=2,sticky="w",padx=5)
        self._slider(tab_manual, 1, "Tilt", -30, 90, self.mv_tilt, 0.5)
        ttk.Entry(tab_manual, textvariable=self.mv_tilt,width=8).grid(row=1,column=2,sticky="w",padx=5)
        self._slider(tab_manual, 2, "Speed", 0, 100, self.mv_speed, 1)
        self._slider(tab_manual, 3, "Accel", 0, 1, self.mv_acc, 0.1)
        Button(tab_manual, text="Center (0,0)", command=self.center).grid(row=4, column=0, sticky="w", pady=4)
        Button(tab_manual, text="Apply Move", command=self.apply_move).grid(row=4, column=1, sticky="e", pady=4)
        self._slider(tab_manual, 5, "LED", 0, 255, self.led, 1)
        Button(tab_manual, text="Set LED", command=self.set_led).grid(row=6, column=1, sticky="e", pady=4)
        Button(tab_manual, text="Laser ON/OFF", command=self.toggle_laser).grid(row=6, column=2, sticky="w", padx=4, pady=4)
    
    def _setup_misc_tab(self):
        for i in range(self.notebook.index("end")):
            if self.notebook.tab(i, "text") == "Preview & Settings":
                tab_misc = self.notebook.nametowidget(self.notebook.tabs()[i])
                break
        
        for widget in tab_misc.winfo_children(): widget.destroy()
        
        misc_sf = ScrollFrame(tab_misc)
        misc_sf.pack(fill="both", expand=True)
        misc = misc_sf.body
        
        row = 0
        Checkbutton(misc, text="Live Preview", variable=self.preview_enable, command=self.toggle_preview).grid(row=row, column=0, sticky="w", pady=2); row += 1
        self._row(misc, row, "Preview w/h/-", self.preview_w, self.preview_h, None, ("W","H","")); row += 1
        self._entry(misc, row, "Preview fps", self.preview_fps); row += 1
        self._entry(misc, row, "Preview quality", self.preview_q); row += 1
        Button(misc, text="Apply Preview Size", command=self.apply_preview_size).grid(row=row, column=1, sticky="w", pady=4); row += 1
        
        ttk.Separator(misc, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=(8,6)); row += 1
        Checkbutton(misc, text="Undistort preview (use calib.npz)", variable=self.ud_enable).grid(row=row, column=0, sticky="w"); row += 1
        Button(misc, text="Load calib.npz", command=self.load_npz).grid(row=row, column=0, sticky="w", pady=2)
        Checkbutton(misc, text="Also save undistorted copy", variable=self.ud_save_copy).grid(row=row, column=1, sticky="w", pady=2); row += 1
        Label(misc, text="Alpha/Balance (0~1)").grid(row=row, column=0, sticky="w")
        Scale(misc, from_=0.0, to=1.0, orient=HORIZONTAL, resolution=0.01, length=200,
              variable=self.ud_alpha, command=lambda v: setattr(self, "_ud_src_size", None)).grid(row=row, column=1, sticky="w"); row += 1
        
        ttk.Separator(misc, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=(8,6)); row += 1
        Label(misc, text="YOLO 가중치 (.pt)").grid(row=row, column=0, sticky="w")
        Button(misc, text="Load YOLO", command=self.load_yolo_weights).grid(row=row, column=1, sticky="w", pady=2); row += 1
        
        for c in range(4): misc.grid_columnconfigure(c, weight=1)

    def _setup_pointing_tab(self):
        """Pointing Tab Layout (Restored from Com_main.py)"""
        for i in range(self.notebook.index("end")):
            if self.notebook.tab(i, "text") == "Pointing":
                self.tab_point = self.notebook.nametowidget(self.notebook.tabs()[i])
                break
        
        for widget in self.tab_point.winfo_children(): widget.destroy()

        # Canvas & Scrollbar
        self.point_canvas = tk.Canvas(self.tab_point)
        self.point_scroll = ttk.Scrollbar(self.tab_point, orient="vertical", command=self.point_canvas.yview)
        self.point_scroll_frame = ttk.Frame(self.point_canvas)
        
        self.point_scroll_frame.bind("<Configure>", lambda e: self.point_canvas.configure(scrollregion=self.point_canvas.bbox("all")))
        self.point_canvas.create_window((0, 0), window=self.point_scroll_frame, anchor="nw")
        self.point_canvas.configure(yscrollcommand=self.point_scroll.set)
        
        # Mouse Wheel
        def _on_mousewheel(event): self.point_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        def _bind_mousewheel(event): self.point_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        def _unbind_mousewheel(event): self.point_canvas.unbind_all("<MouseWheel>")
        self.point_canvas.bind("<Enter>", _bind_mousewheel); self.point_canvas.bind("<Leave>", _unbind_mousewheel)
        self.point_scroll_frame.bind("<Enter>", _bind_mousewheel); self.point_scroll_frame.bind("<Leave>", _unbind_mousewheel)
        
        self.point_canvas.pack(side="left", fill="both", expand=True)
        self.point_scroll.pack(side="right", fill="y")
        
        # Grid Layout
        col1_frame = ttk.Frame(self.point_scroll_frame); col1_frame.grid(row=0, column=0, padx=5, pady=10, sticky="nsew")
        col2_frame = ttk.Frame(self.point_scroll_frame); col2_frame.grid(row=0, column=1, padx=5, pady=10, sticky="nsew")
        col3_frame = ttk.Frame(self.point_scroll_frame); col3_frame.grid(row=0, column=2, padx=5, pady=10, sticky="nsew")
        
        self.point_scroll_frame.grid_columnconfigure(0, weight=1)
        self.point_scroll_frame.grid_columnconfigure(1, weight=1)
        self.point_scroll_frame.grid_columnconfigure(2, weight=1)
        
        # 1. Pointing Settings
        point_set_frame = ttk.LabelFrame(col1_frame, text="Pointing Settings")
        point_set_frame.pack(padx=5, pady=5, fill="both", expand=True)
        def add_entry(parent, label, var, r):
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=var, width=10).grid(row=r, column=1, sticky="w", padx=5, pady=2)
        
        add_entry(point_set_frame, "Laser ROI Size (px):", self.pointing_roi_size, 0)
        ttk.Label(point_set_frame, text="--- Pointing Settings ---").grid(row=1, column=0, columnspan=2, pady=5)
        add_entry(point_set_frame, "Tolerance (px):", self.pointing_px_tol, 2)
        add_entry(point_set_frame, "Min Stable Frames:", self.pointing_min_frames, 3)
        add_entry(point_set_frame, "Max Step (deg):", self.pointing_max_step, 4)
        add_entry(point_set_frame, "LED/LASER Settle (s):", self.point_settle, 5)
        add_entry(point_set_frame, "Loop Interval (s):", self.pointing_interval, 6)

        # 2. Pointing Control
        point_ctrl_frame = ttk.LabelFrame(col2_frame, text="Pointing Control")
        point_ctrl_frame.pack(padx=5, pady=5, fill="both", expand=True)
        ttk.Checkbutton(point_ctrl_frame, text="Enable Pointing Mode", variable=self.pointing_enable, command=self.on_pointing_toggle).pack(anchor="w", padx=5, pady=5)
        
        # 3. CSV Analysis
        point_csv_frame = ttk.LabelFrame(col3_frame, text="CSV Analysis (Legacy)")
        point_csv_frame.pack(padx=5, pady=5, fill="both", expand=True)
        ttk.Label(point_csv_frame, textvariable=self.point_csv_path, wraplength=200).pack(anchor="w", padx=5, pady=2)
        
        ttk.Label(point_csv_frame, text="Conf Min:").pack(anchor="w", padx=5)
        ttk.Entry(point_csv_frame, textvariable=self.point_conf_min, width=15).pack(anchor="w", padx=5)
        ttk.Label(point_csv_frame, text="Min Samples:").pack(anchor="w", padx=5)
        ttk.Entry(point_csv_frame, textvariable=self.point_min_samples, width=15).pack(anchor="w", padx=5)
        
        self.point_result_lbl = ttk.Label(point_csv_frame, text="Result: -")
        self.point_result_lbl.pack(anchor="w", padx=5, pady=5)
        ttk.Button(point_csv_frame, text="Load CSV", command=self.pointing_choose_csv).pack(anchor="w", padx=5, pady=2)
        ttk.Button(point_csv_frame, text="Move to Target", command=self.pointing_move).pack(anchor="w", padx=5, pady=5)
        
        # 4. Tracked Objects (동적 버튼)
        point_objects_frame = ttk.LabelFrame(self.point_scroll_frame, text="🎯 Tracked Objects")
        point_objects_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=10, sticky="nsew")
        
        # 스크롤 가능한 버튼 컨테이너
        self.track_buttons_canvas = tk.Canvas(point_objects_frame, height=150)
        self.track_buttons_scroll = ttk.Scrollbar(point_objects_frame, orient="vertical", command=self.track_buttons_canvas.yview)
        self.track_buttons_frame = ttk.Frame(self.track_buttons_canvas)
        
        self.track_buttons_frame.bind("<Configure>", lambda e: self.track_buttons_canvas.configure(scrollregion=self.track_buttons_canvas.bbox("all")))
        self.track_buttons_canvas.create_window((0, 0), window=self.track_buttons_frame, anchor="nw")
        self.track_buttons_canvas.configure(yscrollcommand=self.track_buttons_scroll.set)
        
        self.track_buttons_canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.track_buttons_scroll.pack(side="right", fill="y")
        
        # 초기 메시지
        ttk.Label(self.track_buttons_frame, text="(스캔 완료 후 자동으로 표시됩니다)", foreground="gray").pack(pady=20)
        
        # Debug Preview
        debug_frame = Frame(self.tab_point, bg="#111", width=420, height=420, relief="solid", borderwidth=2)
        debug_frame.pack(side="right", padx=10, pady=10)
        debug_frame.pack_propagate(False)
        Label(debug_frame, text="Debug Target Preview", bg="#111", fg="white", font=("", 10, "bold")).pack(pady=5)
        self.debug_preview_label = Label(debug_frame, bg="#111", fg="#666", text="(Waiting for detection...)")
        self.debug_preview_label.pack(fill="both", expand=True, padx=10, pady=10)
    
    # ========== PV Monitor Tab 제거됨 (아두이노 전압 측정) ==========
    # _setup_pv_tab() 메서드 및 모든 관련 코드 제거됨
    
    # Helper methods for UI creation
    def _create_target_buttons(self, computed_targets):
        """
        track_id별 "Move to Target" 버튼 동적 생성
        
        Args:
            computed_targets: {track_id: (pan, tilt), ...}
        """
        if not hasattr(self, 'track_buttons_frame'):
            print("[UI] track_buttons_frame not ready yet")
            return
        
        # 기존 버튼 모두 제거
        for widget in self.track_buttons_frame.winfo_children():
            widget.destroy()
        
        if not computed_targets:
            ttk.Label(self.track_buttons_frame, text="(검출된 객체 없음)", foreground="gray").pack(pady=20)
            return
        
        # track_id별 버튼 생성
        for track_id in sorted(computed_targets.keys()):
            pan, tilt = computed_targets[track_id]
            
            btn_frame = ttk.Frame(self.track_buttons_frame)
            btn_frame.pack(fill="x", padx=5, pady=3)
            
            # Track 정보 라벨
            info_text = f"Track {track_id}: Pan={pan}°, Tilt={tilt}°"
            ttk.Label(btn_frame, text=info_text, width=35).pack(side="left", padx=5)
            
            # Move 버튼
            ttk.Button(
                btn_frame, 
                text="Move to Target", 
                command=lambda tid=track_id: self.move_to_target(tid),
                width=15
            ).pack(side="right", padx=5)
        
        print(f"[UI] Created {len(computed_targets)} target button(s)")
    
    def _row(self, parent, r, txt, v1, v2, v3, labels=("Min","Max","Step")):
        Label(parent, text=txt).grid(row=r, column=0, sticky="w", padx=(5,10))
        ttk.Entry(parent, textvariable=v1, width=8).grid(row=r, column=1, sticky="w", padx=2)
        Label(parent, text=labels[0], font=("", 8)).grid(row=r, column=1, sticky="e", padx=2)
        ttk.Entry(parent, textvariable=v2, width=8).grid(row=r, column=2, sticky="w", padx=2)
        Label(parent, text=labels[1], font=("", 8)).grid(row=r, column=2, sticky="e", padx=2)
        if v3 is not None:
            ttk.Entry(parent, textvariable=v3, width=8).grid(row=r, column=3, sticky="w", padx=2)
            Label(parent, text=labels[2], font=("", 8)).grid(row=r, column=3, sticky="e", padx=2)
    
    def _entry(self, parent, r, txt, var):
        Label(parent, text=txt).grid(row=r, column=0, sticky="w", padx=(5,10))
        ttk.Entry(parent, textvariable=var, width=12).grid(row=r, column=1, sticky="w", padx=2)
    
    def _slider(self, parent, r, txt, mi, ma, var, res):
        Label(parent, text=txt).grid(row=r, column=0, sticky="w", padx=5)
        Scale(parent, from_=mi, to=ma, orient=HORIZONTAL, resolution=res, length=250,
              variable=var).grid(row=r, column=1, sticky="w", padx=5)
    
    def _send_snap_cmd(self, save_name):
        self.ctrl.send({
            "cmd": "snap",
            "width": self.width.get(),
            "height": self.height.get(),
            "quality": self.quality.get(),
            "save": save_name,
            "ud_save": self.ud_save_copy.get()
        })
    
    def _undistort_pair(self, img_on, img_off):
        """Undistort a pair of images"""
        self.image_processor.alpha = float(self.ud_alpha.get())
        img_on_ud = self.image_processor.undistort(img_on, use_torch=True)
        img_off_ud = self.image_processor.undistort(img_off, use_torch=True)
        return img_on_ud, img_off_ud
    
    def _get_yolo_model(self):
        path = self.yolo_wpath.get()
        if not path or not pathlib.Path(path).exists():
            return None
        return self.yolo_processor.get_model(path)
    
    def _get_device(self):
        return self.yolo_processor.get_device()
    
    def _calculate_angle_delta(self, err_x, err_y, k_pan=None, k_tilt=None, force_max_step=None):
        """Calculate angle delta from pixel error"""
        # Default gains
        if k_pan is None:
            k_pan = 0.02
        if k_tilt is None:
            k_tilt = 0.02
        
        d_pan = err_x * k_pan
        d_tilt = err_y * k_tilt
        
        # Apply max step limit
        max_step = force_max_step if force_max_step is not None else self.pointing_max_step.get()
        d_pan = max(-max_step, min(max_step, d_pan))
        d_tilt = max(-max_step, min(max_step, d_tilt))
        
        return d_pan, d_tilt

        # Find or create PV tab
        pv_tab = None
        for i in range(self.notebook.index("end")):
            if self.notebook.tab(i, "text") == "PV Monitor":
                pv_tab = self.notebook.nametowidget(self.notebook.tabs()[i])
                break
        
        if pv_tab is None:
            # Tab doesn't exist, will be created in test.py
            return
        
        # Clear existing widgets
        for widget in pv_tab.winfo_children():
            widget.destroy()
        
        # Check if matplotlib is available
        if not MATPLOTLIB_AVAILABLE:
            error_frame = Frame(pv_tab)
            error_frame.pack(fill="both", expand=True, padx=20, pady=20)
            Label(error_frame, text="⚠️ PV Monitor 사용 불가", font=("", 14, "bold"), fg="red").pack(pady=10)
            Label(error_frame, text="matplotlib이 설치되지 않았습니다.", font=("", 11)).pack(pady=5)
            Label(error_frame, text="설치 명령어:", font=("", 10, "bold")).pack(pady=10)
            Label(error_frame, text="pip install matplotlib", font=("Consolas", 10), bg="#f0f0f0").pack(pady=5)
            return
        
        # Main container
        main_frame = Frame(pv_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = Frame(main_frame, relief="ridge", borderwidth=2)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        
        # Title
        Label(left_panel, text="PV Monitor Control", font=("", 12, "bold")).pack(pady=10)
        
        # Port selection
        port_frame = Frame(left_panel)
        port_frame.pack(pady=5, padx=10, fill="x")
        Label(port_frame, text="Port:").pack(side="left")
        ttk.Entry(port_frame, textvariable=self.pv_port, width=10).pack(side="left", padx=5)
        
        # Start/Stop buttons
        btn_frame = Frame(left_panel)
        btn_frame.pack(pady=10, padx=10)
        self.pv_start_btn = Button(btn_frame, text="▶ Start", command=self.start_pv_monitoring, 
                                    bg="#4CAF50", fg="white", width=12)
        self.pv_start_btn.pack(pady=5)
        self.pv_stop_btn = Button(btn_frame, text="■ Stop", command=self.stop_pv_monitoring,
                                   bg="#f44336", fg="white", width=12, state="disabled")
        self.pv_stop_btn.pack(pady=5)
        
        # Clear button
        Button(left_panel, text="Clear Graph", command=self.clear_pv_graph, width=12).pack(pady=5)
        
        # Status
        ttk.Separator(left_panel, orient="horizontal").pack(fill="x", pady=10)
        Label(left_panel, text="Current Values", font=("", 10, "bold")).pack(pady=5)
        
        self.pv_voltage_label = Label(left_panel, text="Voltage: -- V", font=("", 10))
        self.pv_voltage_label.pack(anchor="w", padx=10, pady=2)
        
        self.pv_current_label = Label(left_panel, text="Current: -- mA", font=("", 10))
        self.pv_current_label.pack(anchor="w", padx=10, pady=2)
        
        self.pv_power_label = Label(left_panel, text="Power: -- mW", font=("", 10))
        self.pv_power_label.pack(anchor="w", padx=10, pady=2)
        
        # Status message
        ttk.Separator(left_panel, orient="horizontal").pack(fill="x", pady=10)
        self.pv_status_label = Label(left_panel, text="Status: Ready", fg="gray", wraplength=150)
        self.pv_status_label.pack(anchor="w", padx=10, pady=5)
        
        # Right panel - Graph
        graph_panel = Frame(main_frame, relief="ridge", borderwidth=2)
        graph_panel.pack(side="right", fill="both", expand=True)
        
        # Create matplotlib figure
        self.pv_figure = Figure(figsize=(8, 6), dpi=80)
        
        # Create 3 subplots
        self.pv_ax_voltage = self.pv_figure.add_subplot(311)
        self.pv_ax_voltage.set_ylabel('Voltage (V)', fontsize=9)
        self.pv_ax_voltage.grid(True, alpha=0.3)
        
        self.pv_ax_current = self.pv_figure.add_subplot(312)
        self.pv_ax_current.set_ylabel('Current (mA)', fontsize=9)
        self.pv_ax_current.grid(True, alpha=0.3)
        
        self.pv_ax_power = self.pv_figure.add_subplot(313)
        self.pv_ax_power.set_xlabel('Time (s)', fontsize=9)
        self.pv_ax_power.set_ylabel('Power (mW)', fontsize=9)
        self.pv_ax_power.grid(True, alpha=0.3)
        
        self.pv_figure.tight_layout()
        
        # Embed figure in tkinter
        self.pv_canvas = FigureCanvasTkAgg(self.pv_figure, master=graph_panel)
        self.pv_canvas.draw()
        self.pv_canvas.get_tk_widget().pack(fill="both", expand=True)