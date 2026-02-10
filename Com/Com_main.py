#!/usr/bin/env python3
"""
Com Client - Modular architecture with mixins
"""

import pathlib
import datetime
from tkinter import Tk, Label, Frame, ttk

from network import GuiCtrlClient, GuiImgClient
from event_handlers import EventHandlersMixin
from pointing_handler import PointingHandlerMixin
from app_helpers import AppHelpersMixin
from ui_components import PreviewFrame, ScanTab, TestSettingsTab, PointingTab
from scan_controller import ScanController
from yolo_utils import YOLOProcessor

SERVER_HOST = "127.0.0.1"
GUI_CTRL_PORT = 7600
GUI_IMG_PORT = 7601

# 저장 디렉토리
SAVE_DIR = pathlib.Path("captures")


class ComApp(EventHandlersMixin, PointingHandlerMixin, AppHelpersMixin):
    """메인 앱 - 믹스인 패턴으로 이벤트 처리 분리"""
    
    def __init__(self, root: Tk):
        self.root = root
        root.title("IR-CUT Camera System")
        root.geometry("1200x800")
        
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
        tab_pointing = Frame(self.notebook)
        
        self.notebook.add(tab_scan, text="Scan")
        self.notebook.add(tab_test, text="Test & Settings")
        self.notebook.add(tab_pointing, text="Pointing")
        
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
        
        pointing_callbacks = {
            'pointing_choose_csv': self.pointing_choose_csv,
            'pointing_compute': self.pointing_compute,
            'move_to_target': self.move_to_target
        }
        self.pointing_tab = PointingTab(tab_pointing, pointing_callbacks)
        
        # pointing_handler에서 참조할 수 있도록 변수 연결
        self.point_csv_path = self.pointing_tab.point_csv_path
        self._create_target_buttons = self.pointing_tab._create_target_buttons
        
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
        
        # 레이저 상태
        self.laser_state = False
        
        # Preview 상태 추적
        self.preview_active = False
        
        # YOLO Processor
        self.yolo_processor = YOLOProcessor()
        
        # Scan Controller
        self.scan_ctrl = ScanController(SAVE_DIR, self.yolo_processor)
        
        # Frame count (for preview)
        self.frame_count = 0
        
        # Network Clients
        self.ctrl = GuiCtrlClient(SERVER_HOST, GUI_CTRL_PORT)
        self.ctrl.start()
        
        self.img = GuiImgClient(SERVER_HOST, GUI_IMG_PORT, SAVE_DIR)
        self.img.start()
        
        # Start polling (from EventHandlersMixin)
        self.root.after(100, self._init_raspberrypi)
        self.root.after(50, self._poll)
    
    def _init_raspberrypi(self):
        """Raspberrypi 초기 상태 설정 (모든 하드웨어 초기화)"""
        print("[INIT] Raspberrypi 전체 초기화...")
        
        # 1. Preview OFF
        self.ctrl.send({"cmd": "preview", "enable": False})
        
        # 2. LED OFF (0)
        self.ctrl.send({"cmd": "led", "value": 0})
        
        # 3. Laser OFF
        self.ctrl.send({"cmd": "laser", "value": 0})
        self.laser_state = False
        
        # 4. Pan/Tilt Center (0, 0)
        self.ctrl.send({
            "cmd": "move",
            "pan": 0.0,
            "tilt": 0.0,
            "speed": 100,
            "acc": 1.0
        })
        
        # 5. IR-CUT Normal Mode (가시광선)
        self.ctrl.send({"cmd": "ir_cut", "mode": "night"})
        
        print("[INIT] ✅ 초기화 완료: Preview=OFF, LED=0, Laser=OFF, Pan/Tilt=0,0, IR-CUT=Normal")
    
    # ========== Scan Callbacks ==========
    def start_scan(self, params):
        """스캔 시작"""
        # ⭐ 버튼 상태 변경 (Start -> Disabled, Stop -> Normal)
        self.scan_tab.set_scan_state(True)
        
        # ⭐ Preview가 켜져있으면 자동 중지
        if self.preview_active:
            print("[SCAN] Preview 자동 중지...")
            self.toggle_preview(False, 640, 480, 10, 80)
            self.preview_active = False
            # Preview 완전히 중지될 때까지 대기
            self.root.after(300)
        
        # YOLO weights 경로 추출
        yolo_weights = params.pop('yolo_weights', None)
        if yolo_weights and not yolo_weights.strip():
            yolo_weights = None
        
        # ScanController로 세션 시작
        session = self.scan_ctrl.start_session(yolo_weights_path=yolo_weights)
        
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
        self.ctrl.send(cmd)
        self.info_label.config(text=f"🔄 스캔 시작: {session}")
    
    def stop_scan(self):
        """스캔 중지"""
        print(f"[SCAN] Stop")
        self.ctrl.send({"cmd": "scan_stop"})
        result = self.scan_ctrl.stop_session()  # 이제 딕셔너리 반환
        
        # UI 업데이트
        if result:
            self.info_label.config(text=f"⏹️ 스캔 중지: {result['done']}/{result['total']}")
            self.scan_tab.set_scan_state(False)
            
            # Pointing 자동 실행
            csv_path = result.get('csv_path_abs')
            if csv_path:
                print(f"[ComApp] Auto-computing pointing for: {csv_path}")
                # Pointing 탭으로 전환 (선택 사항)
                self.notebook.select(2)  # Pointing Tab index
                self.pointing_compute(csv_path)
            else:
                print("[ComApp] No CSV path returned for auto-pointing")
        else:
             self.info_label.config(text="⏹️ 스캔 중지 (No result)")
             self.scan_tab.set_scan_state(False)
    
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
        self.ctrl.send(cmd)
        self.info_label.config(text=f"🎯 이동: Pan={pan}°, Tilt={tilt}°")
    
    def set_led(self, value):
        """LED 설정"""
        print(f"[LED] Value={value}")
        cmd = {
            "cmd": "led",
            "value": value
        }
        self.ctrl.send(cmd)
        self.info_label.config(text=f"💡 LED: {value}")
    
    def toggle_laser(self):
        """레이저 토글"""
        self.laser_state = not self.laser_state
        print(f"[LASER] Toggle → {self.laser_state}")
        
        cmd = {
            "cmd": "laser",
            "value": 1 if self.laser_state else 0
        }
        self.ctrl.send(cmd)
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
        self.ctrl.send(cmd)
        
        # ⭐ Preview 상태 추적
        self.preview_active = enable
        
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
        self.ctrl.send(cmd)
        
        # 실제 하드웨어 동작: day=IR통과, night=가시광선
        if mode == "night":  # Normal 버튼
            self.info_label.config(text="🔍 Normal Mode (가시광선)")
        else:  # day → IR Mode 버튼
            self.info_label.config(text="🔴 IR Mode (적외선)")
    
    def snap_capture(self):
        """Snap 캡처 - Preview 해상도 사용"""
        w = self.test_tab.preview_w.get()
        h = self.test_tab.preview_h.get()
        print(f"[SNAP] Capturing {w}x{h}")
        
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
        self.ctrl.send(cmd)
        
        self.info_label.config(text=f"📸 캡처 중... ({w}x{h})")
    
    def run(self):
        self.root.mainloop()


def main():
    root = Tk()
    ComApp(root).run()


if __name__ == "__main__":
    main()
