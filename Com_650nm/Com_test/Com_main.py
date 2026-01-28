#!/usr/bin/env python3
# pc_gui.py — GUI client connecting to pc_server.py (not to Pi agent)

import json, socket, struct, threading, queue, pathlib, io
from datetime import datetime
from tkinter import Tk, Label, Button, Scale, HORIZONTAL, IntVar, DoubleVar, Frame, Checkbutton, BooleanVar, filedialog, StringVar
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
import os, re, csv, time
import numpy as np
import cv2

# Import from modules
from network import GuiCtrlClient, GuiImgClient, ui_q
from image_utils import ImageProcessor
from yolo_utils import YOLOProcessor
from scan_utils import ScanController
from app_helpers import AppHelpersMixin
from pointing_handler import PointingHandlerMixin
from event_handlers import EventHandlersMixin
from app_ui import AppUIMixin
# PVMonitor 제거됨 (아두이노 전압 측정)

SERVER_HOST = "127.0.0.1"
GUI_CTRL_PORT = 7600
GUI_IMG_PORT  = 7601
DEFAULT_OUT_DIR = pathlib.Path(f"captures_gui_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
POLL_INTERVAL_MS = 60

class App(AppHelpersMixin, PointingHandlerMixin, EventHandlersMixin, AppUIMixin):
    def __init__(self, root: Tk):
        self.root = root
        root.title("Pan-Tilt Socket GUI (Client)")
        root.geometry("980x820")
        root.minsize(980, 820)

        # connections
        self.ctrl = GuiCtrlClient(SERVER_HOST, GUI_CTRL_PORT); self.ctrl.start()
        self.img  = GuiImgClient (SERVER_HOST, GUI_IMG_PORT, DEFAULT_OUT_DIR); self.img.start()

        # state
        self.tkimg=None
        self._resume_preview_after_snap = False

        # Processors
        self.image_processor = ImageProcessor()
        self.yolo_processor = YOLOProcessor()
        # ⭐ MOT 파라미터 조정 가능 (선택적):
        # self.scan_controller = ScanController(
        #     self.image_processor, self.yolo_processor, DEFAULT_OUT_DIR,
        #     mot_roi_size=300,      # ROI 크기 (픽셀)
        #     mot_grid_size=(11, 11) # 격자 크기 (rows, cols)
        # )
        self.scan_controller = ScanController(self.image_processor, self.yolo_processor, DEFAULT_OUT_DIR)
        print(f"[INFO] cv2.cuda={self.image_processor._use_cv2_cuda}, torch_cuda={self.image_processor._torch_cuda}")

        # ==== Variables (변수 선언은 유지) ====
        self.outdir = StringVar(value=str(DEFAULT_OUT_DIR))
        self.ud_enable    = BooleanVar(value=True)
        self.ud_save_copy = BooleanVar(value=True)
        self.ud_alpha     = DoubleVar(value=0.0)
        self.yolo_wpath = StringVar(value="yolov11m_diff.pt")
        
        # Scan params
        self.pan_min=IntVar(value=-180); self.pan_max=IntVar(value=180); self.pan_step=IntVar(value=15)
        self.tilt_min=IntVar(value=-30); self.tilt_max=IntVar(value=90);  self.tilt_step=IntVar(value=15)
        self.width=IntVar(value=2592);   self.height=IntVar(value=1944); self.quality=IntVar(value=90)
        self.speed=IntVar(value=0);    self.acc=DoubleVar(value=0.0);  self.settle=DoubleVar(value=0.1)
        self.led_settle=DoubleVar(value=0.4)

        # Pointing variables
        self.point_csv_path = StringVar(value="")
        self.point_conf_min = DoubleVar(value=0.50)
        self.point_min_samples = IntVar(value=2)
        self.point_pan_target  = DoubleVar(value=0.0)
        self.point_tilt_target = DoubleVar(value=0.0)
        self.point_speed  = IntVar(value=self.speed.get())
        self.point_acc    = DoubleVar(value=self.acc.get())
        self.pointing_roi_size = IntVar(value=200)
        self.pointing_px_tol = IntVar(value=7)
        self.pointing_min_frames = IntVar(value=4)
        self.pointing_max_step = DoubleVar(value=5.0)
        self.point_settle = DoubleVar(value=0.3)
        self.pointing_enable = BooleanVar(value=False)
        self.pointing_interval = DoubleVar(value=0.1)

        # Manual/Preview variables
        self.mv_pan=DoubleVar(value=0.0); self.mv_tilt=DoubleVar(value=0.0)
        self.mv_speed=IntVar(value=100);  self.mv_acc=DoubleVar(value=1.0)
        self.led=IntVar(value=0)
        self.laser_on = BooleanVar(value=False)
        self.preview_enable=BooleanVar(value=True)
        self.preview_w=IntVar(value=2592); self.preview_h=IntVar(value=1944)
        self.preview_fps=IntVar(value=5); self.preview_q=IntVar(value=70)

        # Internal State
        self._scan_csv_path = None
        self._pointing_log_fp = None
        self._pointing_log_writer = None
        self._pointing_logging = False
        self._pointing_state = 0
        self._pointing_last_ts = 0
        self._pointing_stable_cnt = 0
        self._curr_pan = 0.0; self._curr_tilt = 0.0
        self._fits_h = {}; self._fits_v = {}
        
        # PV Monitor 제거됨 (아두이노 전압 측정)
        # self.pv_port = StringVar(value="COM8")
        # self.pv_monitoring = BooleanVar(value=False)
        # self.pv_monitor = PVMonitor(max_history=100, output_dir=DEFAULT_OUT_DIR)

        # ==== UI Layout (뼈대만 생성) ====
        # 1. Top Bar
        top = Frame(root); top.pack(fill="x", padx=10, pady=6)
        Button(top, text="한장 찍기 (Snap)", command=self.snap_one).pack(side="left", padx=(0,8))
        Button(top, text="출력 폴더", command=self.choose_outdir).pack(side="right")

        # 2. Preview Box
        center = Frame(root); center.pack(fill="x", padx=10)
        self.PREV_W, self.PREV_H = 800, 450
        self.preview_box = Frame(center, width=self.PREV_W, height=self.PREV_H,
                                 bg="#111", highlightthickness=1, highlightbackground="#333")
        self.preview_box.pack()
        self.preview_box.pack_propagate(False)
        self.preview_label = Label(self.preview_box, bg="#111")
        self.preview_label.place(x=0, y=0, width=self.PREV_W, height=self.PREV_H)

        # 3. Tabs (Notebook)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="x", padx=10, pady=(6,10))
        
        # 빈 탭 껍데기만 만들어 둡니다 (app_ui.py가 여기를 채웁니다)
        self.notebook.add(Frame(self.notebook), text="Scan")
        self.notebook.add(Frame(self.notebook), text="Manual / LED")
        self.notebook.add(Frame(self.notebook), text="Preview & Settings")
        self.notebook.add(Frame(self.notebook), text="Pointing")
        # PV Monitor 탭 제거됨

        # ==== UI Setup 호출 (여기가 진짜!) ====
        # app_ui.py의 setup_ui()가 위에서 만든 변수들을 가지고 화면을 채웁니다.
        self.setup_ui() 

        # Auto-load
        if pathlib.Path("calib.npz").exists(): self.load_npz("calib.npz")
        if pathlib.Path(self.yolo_wpath.get()).exists(): self._get_yolo_model()

        self.root.after(POLL_INTERVAL_MS, self._poll)
        # [추가] 0.5초 뒤에 강제로 프리뷰 시작 명령 보내기 (이게 해결책!)
        self.root.after(500, self.resume_preview)

    def _poll(self):
        """
        [Override] Main event loop with throttling to prevent UI freeze
        """
        self._check_pointing_trigger()
        
        # PV status 체크 제거됨 (아두이노)
        
        # Limit processing to 20 events per cycle to keep UI responsive
        for _ in range(20):
            try:
                tag, payload = ui_q.get_nowait()
                
                if tag == "evt":
                    self._handle_server_event(payload)
                elif tag == "preview":
                    self._set_preview(payload)
                elif tag == "saved":
                    self._handle_saved_image(payload)
                elif tag == "toast":
                    print(f"[TOAST] {payload}")
                elif tag == "pointing_step_2":
                    self._handle_pointing_step2()
                elif tag == "preview_on":
                    self._handle_preview_on()
                elif tag == "debug_preview":
                    self._update_debug_preview(payload)
            except queue.Empty:
                break
        
        self.root.after(POLL_INTERVAL_MS, self._poll)

    # ... (run, load_npz 등 메서드들은 그대로 유지하되 UI 생성 코드만 없으면 됨)
    def run(self): self.root.mainloop()

    def load_npz(self, path=None):
        if path is None: path = filedialog.askopenfilename(filetypes=[("NPZ","*.npz")])
        if not path: return
        self.image_processor.alpha = float(self.ud_alpha.get())
        if self.image_processor.load_calibration(path): print(f"[App] Calibration loaded")

    def _undistort_bgr(self, bgr):
        self.image_processor.alpha = float(self.ud_alpha.get())
        return self.image_processor.undistort(bgr, use_torch=True)

    def resume_preview(self):
        if not self.ctrl.sock:
            print("[App] Control not connected yet, retrying resume_preview in 500ms...")
            self.root.after(500, self.resume_preview)
            return

        if self.preview_enable.get():
            self.ctrl.send({"cmd":"preview", "enable": True, "width": self.preview_w.get(), "height": self.preview_h.get(), "fps": self.preview_fps.get(), "quality":self.preview_q.get()})

    def choose_outdir(self):
        d = filedialog.askdirectory()
        if d:
            global DEFAULT_OUT_DIR
            DEFAULT_OUT_DIR = pathlib.Path(d)
            self.outdir.set(str(DEFAULT_OUT_DIR)) # Update String Var

    def load_yolo_weights(self):
        path = filedialog.askopenfilename(filetypes=[("YOLO weights", "*.pt"), ("All files", "*.*")])
        if path:
            self.yolo_wpath.set(path)
            ui_q.put(("toast", f"YOLO 가중치 로드: {pathlib.Path(path).name}"))

    def start_scan(self):
        if not self.image_processor.has_calibration():
            ui_q.put(("toast", "❌ 보정 필요 (Load calib.npz)"))
            return
        if self.preview_enable.get(): self.ctrl.send({"cmd":"preview","enable": False})
        self.ctrl.send({
            "cmd":"scan_run",
            "pan_min":self.pan_min.get(),"pan_max":self.pan_max.get(),"pan_step":self.pan_step.get(),
            "tilt_min":self.tilt_min.get(),"tilt_max":self.tilt_max.get(),"tilt_step":self.tilt_step.get(),
            "speed":self.speed.get(),"acc":float(self.acc.get()),"settle":float(self.settle.get()),
            "led_settle":float(self.led_settle.get()),
            "width":self.width.get(),"height":self.height.get(),"quality":self.quality.get(),
            "session":datetime.now().strftime("scan_%Y%m%d_%H%M%S")
        })

    def stop_scan(self):
        self.ctrl.send({"cmd":"scan_stop"})
        result = self.scan_controller.stop_scan()
        if result and result.get('csv_path'):
            self.point_csv_path.set(str(result['csv_path']))
            ui_q.put(("toast", f"✅ Scan 완료! CSV 로드됨"))
            self.pointing_compute()
        self.root.after(500, lambda: ui_q.put(("preview_on", None)))

    # Pointing / Manual Helpers
    def center(self): self.ctrl.send({"cmd":"move","pan":0.0,"tilt":0.0,"speed":self.speed.get(),"acc":float(self.acc.get())})
    def apply_move(self): self.ctrl.send({"cmd":"move","pan":float(self.mv_pan.get()),"tilt":float(self.mv_tilt.get()),"speed":self.mv_speed.get(),"acc":float(self.mv_acc.get())})
    def set_led(self): self.ctrl.send({"cmd":"led","value":int(self.led.get())})
    def toggle_laser(self):
        val = 1 if not self.laser_on.get() else 0
        self.laser_on.set(bool(val))
        self.ctrl.send({"cmd":"laser", "value": val})

    def toggle_preview(self):
        if self.preview_enable.get():
            self.ctrl.send({"cmd":"preview","enable": True, "width": self.preview_w.get(), "height": self.preview_h.get(), "fps": self.preview_fps.get(), "quality": self.preview_q.get()})
        else:
            self.ctrl.send({"cmd":"preview","enable": False})

    def apply_preview_size(self):
        w = max(160, min(2592, self.preview_w.get()))
        h = max(120,  min(1944, self.preview_h.get()))
        self.preview_w.set(w); self.preview_h.set(h)
        self.toggle_preview()

    def snap_one(self):
        self._resume_preview_after_snap = False
        if self.preview_enable.get():
            self.ctrl.send({"cmd":"preview","enable": False})
            self._resume_preview_after_snap = True
        fname = datetime.now().strftime("snap_%Y%m%d_%H%M%S.jpg")
        self._send_snap_cmd(fname)
    
    # ========== PV Monitoring Methods 제거됨 (아두이노 전압 측정) ==========
    # 아래 메서드들은 아두이노 시리얼 통신을 통한 전압/전류 측정에 사용되었으나 제거됨:
    # - start_pv_monitoring()
    # - stop_pv_monitoring()  
    # - clear_pv_graph()
    # - update_pv_graph()

def main():
    root = Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()