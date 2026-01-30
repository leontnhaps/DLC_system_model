#!/usr/bin/env python3
"""
Com Client - Complete UI with tabs
"""

import queue
import pathlib
import datetime
from tkinter import Tk, Label, Frame, ttk, messagebox
from network import GuiCtrlClient, GuiImgClient
from ui_components import PreviewFrame, ScanTab, ManualTab, PreviewTab

SERVER_HOST = "127.0.0.1"
GUI_CTRL_PORT = 7600
GUI_IMG_PORT = 7601

# 저장 디렉토리
SAVE_DIR = pathlib.Path("captures")

class App:
    def __init__(self, root: Tk):
        self.root = root
        root.title("IR-CUT Camera System")
        root.geometry("1200x800")
        
        # 이미지 큐
        self.img_queue = queue.Queue()
        
        # Main Layout: Left=Tabs, Right=Preview
        main_container = Frame(root)
        main_container.pack(fill="both", expand=True)
        
        # Left Panel: Tabs
        left_panel = Frame(main_container, width=500)
        left_panel.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # Notebook (Tabs)
        self.notebook = ttk.Notebook(left_panel)
        self.notebook.pack(fill="both", expand=True)
        
        # Create Tabs
        tab_scan = Frame(self.notebook)
        tab_manual = Frame(self.notebook)
        tab_preview = Frame(self.notebook)
        
        self.notebook.add(tab_scan, text="Scan")
        self.notebook.add(tab_manual, text="Manual / LED")
        self.notebook.add(tab_preview, text="Preview & Settings")
        
        # Initialize Tab Content
        scan_callbacks = {
            'start_scan': self.start_scan,
            'stop_scan': self.stop_scan
        }
        self.scan_tab = ScanTab(tab_scan, scan_callbacks)
        
        manual_callbacks = {
            'apply_move': self.apply_move,
            'set_led': self.set_led,
            'toggle_laser': self.toggle_laser
        }
        self.manual_tab = ManualTab(tab_manual, manual_callbacks)
        
        preview_callbacks = {
            'toggle_preview': self.toggle_preview,
            'set_ir_cut': self.set_ir_cut,
            'snap_capture': self.snap_capture
        }
        self.preview_tab = PreviewTab(tab_preview, preview_callbacks)
        
        # Right Panel: Preview
        right_panel = Frame(main_container)
        right_panel.pack(side="right", fill="both", padx=5, pady=5)
        
        Label(right_panel, text="📹 Live Preview", font=("", 12, "bold")).pack(pady=5)
        self.preview_frame = PreviewFrame(right_panel, width=640, height=480)
        
        # Info Label
        self.info_label = Label(right_panel, text="연결 대기 중...", font=("", 10))
        self.info_label.pack(pady=10)
        
        # 저장 디렉토리 생성
        SAVE_DIR.mkdir(exist_ok=True)
        
        # 저장 대기 이미지 (snap용)
        self.snap_images = []  # [(name, data), ...]
        
        # 레이저 상태
        self.laser_state = False
        
        # Network Clients
        self.ctrl_client = GuiCtrlClient(SERVER_HOST, GUI_CTRL_PORT)
        self.ctrl_client.start()
        
        self.img_client = GuiImgClient(SERVER_HOST, GUI_IMG_PORT, self.img_queue)
        self.img_client.start()
        
        # Polling
        self.frame_count = 0
        self.root.after(50, self._poll)
    
    # ========== Scan Callbacks ==========
    def start_scan(self, params):
        """스캔 시작"""
        print(f"[SCAN] Start: {params}")
        cmd = {
            "cmd": "scan_run",
            **params
        }
        self.ctrl_client.send(cmd)
        self.info_label.config(text="🔄 스캔 시작...")
    
    def stop_scan(self):
        """스캔 중지"""
        print(f"[SCAN] Stop")
        self.ctrl_client.send({"cmd": "scan_stop"})
        self.info_label.config(text="⏹️ 스캔 중지")
    
    # ========== Manual Callbacks ==========
    def apply_move(self, pan, tilt, speed, acc):
        """Pan/Tilt 이동"""
        print(f"[MOVE] Pan={pan}, Tilt={tilt}, Speed={speed}, Acc={acc}")
        cmd = {
            "cmd": "move",
            "pan": pan,
            "tilt": tilt,
            "speed": speed,
            "acc": acc
        }
        self.ctrl_client.send(cmd)
        self.info_label.config(text=f"🎯 이동: Pan={pan}°, Tilt={tilt}°")
    
    def set_led(self, value):
        """LED 설정"""
        print(f"[LED] Value={value}")
        cmd = {
            "cmd": "led",
            "value": value
        }
        self.ctrl_client.send(cmd)
        self.info_label.config(text=f"💡 LED: {value}")
    
    def toggle_laser(self):
        """레이저 토글"""
        self.laser_state = not self.laser_state
        print(f"[LASER] Toggle → {self.laser_state}")
        
        cmd = {
            "cmd": "laser",
            "value": 1 if self.laser_state else 0
        }
        self.ctrl_client.send(cmd)
        self.info_label.config(text=f"🔴 레이저: {'ON' if self.laser_state else 'OFF'}")
    
    # ========== Preview Callbacks ==========
    def toggle_preview(self, enable, w, h, fps, q):
        """프리뷰 토글"""
        print(f"[PREVIEW] Enable={enable}, {w}x{h} @ {fps}fps")
        cmd = {
            "cmd": "preview",
            "enable": enable,
            "width": w,
            "height": h,
            "fps": fps,
            "quality": q
        }
        self.ctrl_client.send(cmd)
        
        if enable:
            self.info_label.config(text=f"✅ 프리뷰: {w}x{h}")
        else:
            self.info_label.config(text="⏸️ 프리뷰 중지")
    
    def set_ir_cut(self, mode):
        """IR-CUT 모드 설정"""
        print(f"[IR-CUT] Mode={mode}")
        cmd = {
            "cmd": "ir_cut",
            "mode": mode
        }
        self.ctrl_client.send(cmd)
        
        if mode == "day":
            self.info_label.config(text="☀️ Day Mode (IR 필터 ON)")
        else:
            self.info_label.config(text="🌙 Night Mode (IR 필터 OFF)")
    
    def snap_capture(self):
        """Snap 캡처 - 5MP + 3MP (순차 촬영)"""
        print(f"[SNAP] Capturing 5MP + 3MP")
        
        # 저장 이미지 초기화
        self.snap_images = []
        
        # 타임스탬프
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 5MP (2592x1944) - 최대 해상도
        cmd_5mp = {
            "cmd": "snap",
            "width": 2592,
            "height": 1944,
            "quality": 95,
            "save": f"snap_{ts}_5MP.jpg"
        }
        self.ctrl_client.send(cmd_5mp)
        
        # 딜레이 후 3MP 촬영 (순차 처리)
        self.root.after(1500, lambda: self._snap_second(ts))
        
        self.info_label.config(text="📸 캡처 중... (5MP)")
    
    def _snap_second(self, ts):
        """두 번째 snap (3MP)"""
        cmd_3mp = {
            "cmd": "snap",
            "width": 2048,
            "height": 1536,
            "quality": 95,
            "save": f"snap_{ts}_3MP.jpg"
        }
        self.ctrl_client.send(cmd_3mp)
        self.info_label.config(text="📸 캡처 중... (3MP)")
    
    # ========== Polling ==========
    def _poll(self):
        """이미지 수신 체크"""
        try:
            msg = self.img_queue.get_nowait()
            if len(msg) == 3:
                tag, name, payload = msg
            else:
                tag, payload = msg
                name = None
            
            if tag == "img":
                # Snap 이미지 저장
                if name and not name.startswith("_preview_"):
                    save_path = SAVE_DIR / name
                    with open(save_path, 'wb') as f:
                        f.write(payload)
                    print(f"[SAVE] {save_path}")
                    self.info_label.config(text=f"💾 저장됨: {name}")
                
                # 프리뷰 표시
                self.preview_frame.display_image(payload)
                self.frame_count += 1
        except queue.Empty:
            pass
        
        self.root.after(50, self._poll)
    
    def run(self):
        self.root.mainloop()

def main():
    root = Tk()
    App(root).run()

if __name__ == "__main__":
    main()
