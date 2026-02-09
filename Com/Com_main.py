#!/usr/bin/env python3
"""
Com Client - Complete UI with tabs
"""

import queue
import pathlib
import datetime
from tkinter import Tk, Label, Frame, ttk, messagebox
from network import GuiCtrlClient, GuiImgClient
from ui_components import PreviewFrame, ScanTab, TestSettingsTab
from scan_controller import ScanController

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
        
        # 이벤트 큐 (progress 등)
        self.event_queue = queue.Queue()
        
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
        tab_test = Frame(self.notebook)
        
        self.notebook.add(tab_scan, text="Scan")
        self.notebook.add(tab_test, text="Test & Settings")
        
        # Initialize Tab Content
        scan_callbacks = {
            'start_scan': self.start_scan,
            'stop_scan': self.stop_scan
        }
        self.scan_tab = ScanTab(tab_scan, scan_callbacks)
        
        test_callbacks = {
            'apply_move': self.apply_move,
            'set_led': self.set_led,
            'toggle_laser': self.toggle_laser,
            'toggle_preview': self.toggle_preview,
            'set_ir_cut': self.set_ir_cut,
            'snap_capture': self.snap_capture
        }
        self.test_tab = TestSettingsTab(tab_test, test_callbacks)
        
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
        
        # Scan Controller
        self.scan_ctrl = ScanController(SAVE_DIR)
        
        # Network Clients
        self.ctrl_client = GuiCtrlClient(SERVER_HOST, GUI_CTRL_PORT, self.event_queue)
        self.ctrl_client.start()
        
        self.img_client = GuiImgClient(SERVER_HOST, GUI_IMG_PORT, self.img_queue)
        self.img_client.start()
        
        # Polling
        self.frame_count = 0
        self.root.after(50, self._poll)
        
        # ⚠️ Preview 자동 시작 제거 - 사용자가 직접 켜도록
        # Raspberrypi 초기화 - 이전 실행 상태 초기화
        self.root.after(100, self._init_raspberrypi)
    
    def _init_raspberrypi(self):
        """Raspberrypi 초기 상태 설정 (모든 하드웨어 초기화)"""
        print("[INIT] Raspberrypi 전체 초기화...")
        
        # 1. Preview OFF
        self.ctrl_client.send({"cmd": "preview", "enable": False})
        
        # 2. LED OFF (0)
        self.ctrl_client.send({"cmd": "led", "value": 0})
        
        # 3. Laser OFF
        self.ctrl_client.send({"cmd": "laser", "value": 0})
        self.laser_state = False
        
        # 4. Pan/Tilt Center (0, 0)
        self.ctrl_client.send({
            "cmd": "move",
            "pan": 0.0,
            "tilt": 0.0,
            "speed": 100,
            "acc": 1.0
        })
        
        # 5. IR-CUT Day Mode (필터 ON)
        self.ctrl_client.send({"cmd": "ir_cut", "mode": "day"})
        
        print("[INIT] ✅ 초기화 완료: Preview=OFF, LED=0, Laser=OFF, Pan/Tilt=0,0, IR-CUT=Day")
    
    # ========== Scan Callbacks ==========
    def start_scan(self, params):
        """스캔 시작"""
        # ScanController로 세션 시작
        session = self.scan_ctrl.start_session()
        
        # Progress UI 초기화
        self.scan_tab.prog.configure(value=0, maximum=100)
        self.scan_tab.prog_lbl.config(text="0 / 0")
        
        print(f"[SCAN] Start: {params}")
        
        # Command 전송 (session 이름 포함)
        cmd = {
            "cmd": "scan_run",
            "session": session,
            **params
        }
        self.ctrl_client.send(cmd)
        self.info_label.config(text=f"🔄 스캔 시작: {session}")
    
    def stop_scan(self):
        """스캔 중지"""
        print(f"[SCAN] Stop")
        self.ctrl_client.send({"cmd": "scan_stop"})
        result = self.scan_ctrl.stop_session()
        self.info_label.config(text=f"⏹️ 스캔 중지: {result['done']}/{result['total']}")
    
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
    
    def _restart_preview(self):
        """Preview 재시작 헬퍼"""
        w = self.test_tab.preview_w.get()
        h = self.test_tab.preview_h.get()
        fps = self.test_tab.preview_fps.get()
        q = self.test_tab.preview_q.get()
        
        self.ctrl_client.send({
            "cmd": "preview",
            "enable": True,
            "width": w,
            "height": h,
            "fps": fps,
            "quality": q
        })
    
    def snap_capture(self):
        """Snap 캡처 - Preview 해상도 사용"""
        w = self.test_tab.preview_w.get()
        h = self.test_tab.preview_h.get()
        print(f"[SNAP] Capturing {w}x{h}")
        
        # 저장 이미지 초기화
        self.snap_images = []
        
        # 타임스탬프
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Snap 캡처
        cmd = {
            "cmd": "snap",
            "width": w,
            "height": h,
            "quality": 95,
            "save": f"snap_{ts}.jpg"
        }
        self.ctrl_client.send(cmd)
        
        self.info_label.config(text=f"📸 캡처 중... ({w}x{h})")

    
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
                # ⭐ Scan 이미지 자동 저장 (ScanController 사용)
                if name and self.scan_ctrl.is_active():
                    saved_path = self.scan_ctrl.save_image(name, payload)
                    if saved_path:
                        # Scan 이미지는 세션 폴더에 저장됨
                        print(f"[SCAN_SAVE] {saved_path}")
                
                # Snap 이미지 저장 (scan이 아닌 경우만)
                elif name and not name.startswith("_preview_"):
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
        
        # 이벤트 수신 체크 (progress 등)
        try:
            msg = self.event_queue.get_nowait()
            if len(msg) == 2:
                tag, event = msg
                if tag == "event":
                    self._handle_event(event)
        except queue.Empty:
            pass
        
        self.root.after(50, self._poll)
    
    def _handle_event(self, event):
        """스캔 이벤트 처리"""
        evt = event.get("event")
        
        if evt == "start":
            total = event.get("total", 0)
            self.scan_ctrl.update_progress(0, total)
            self.scan_tab.prog.configure(value=0, maximum=total)
            self.scan_tab.prog_lbl.config(text=f"0 / {total}")
            print(f"[EVENT] Scan started: {total} images")
        
        elif evt == "progress":
            done = event.get("done", 0)
            total = event.get("total", 100)
            name = event.get("name", "")
            
            self.scan_ctrl.update_progress(done, total)
            self.scan_tab.prog.configure(value=done, maximum=total)
            self.scan_tab.prog_lbl.config(text=f"{done} / {total}")
            print(f"[EVENT] Progress: {done}/{total} - {name}")
        
        elif evt == "done":
            done, total = self.scan_ctrl.get_progress()
            print(f"[EVENT] Scan completed: {done}/{total}")
            self.info_label.config(text=f"✅ 스캔 완료: {done}/{total}")
        
        elif evt == "error":
            msg = event.get("message", "Unknown error")
            print(f"[EVENT] Error: {msg}")
            self.info_label.config(text=f"❌ 오류: {msg}")
            messagebox.showerror("Scan Error", msg)
    
    def run(self):
        self.root.mainloop()

def main():
    root = Tk()
    App(root).run()

if __name__ == "__main__":
    main()
