#!/usr/bin/env python3
"""
Com Client - Modular architecture with mixins
"""

import pathlib
import datetime
import time
from tkinter import Tk, Label, Frame, ttk

from network import GuiCtrlClient, GuiImgClient
from event_handlers import EventHandlersMixin
from pointing_handler import PointingHandlerMixin
from app_helpers import AppHelpersMixin
from ui_components import PreviewFrame, ScanTab, TestSettingsTab, PointingTab
from scan_controller import ScanController
from yolo_utils import YOLOProcessor
import threading

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
            'move_to_target': self.move_to_target,
            'stop_aiming': self.stop_aiming
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
        
        # YOLO Processor (scan/pointing 공유 인스턴스)
        try:
            shared_yolo = YOLOProcessor()
            self.yolo = shared_yolo
            self.yolo_processor = shared_yolo
            print("[ComApp] YOLO 모델 로드 완료")
        except Exception as e:
            print(f"[ComApp] YOLO 로드 실패 (무시 가능): {e}")
            self.yolo = None  # 없어도 앱은 실행되게
            self.yolo_processor = None

        # 레이저 상태
        self.laser_state = False
        
        # Preview 상태 추적
        self.preview_active = False
        self._resume_preview_after_scan = False
        self._resume_preview_after_snap = False
        self._scan_preview_cfg = None
        self._snap_preview_cfg = None
        self._snap_restore_token = 0
        self._scan_done_pending = False
        self._scan_finalize_idle_s = 1.2
        self._last_scan_image_ts = 0.0
        
        # Scan Controller
        self.scan_ctrl = ScanController(SAVE_DIR, self.yolo_processor)

        # Pointing related initializations (from PointingHandlerMixin)
        self._pointing_gains = {}
        self._pointing_img_event = threading.Event()
        
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
    def _get_preview_cfg(self):
        """현재 Preview UI 설정을 (w, h, fps, q)로 반환"""
        return (
            int(self.test_tab.preview_w.get()),
            int(self.test_tab.preview_h.get()),
            int(self.test_tab.preview_fps.get()),
            int(self.test_tab.preview_q.get())
        )

    def _restore_preview(self, cfg, reason="restore"):
        """저장된 설정으로 Preview 복구"""
        if cfg is None:
            cfg = self._get_preview_cfg()
        w, h, fps, q = cfg
        print(f"[PREVIEW] Auto-restore ({reason}): {w}x{h} @ {fps}fps")
        self.toggle_preview(True, w, h, fps, q)

    def _send_scan_run(self, cmd, session):
        """scan_run 실제 전송 (after 지연 전송용 분리)"""
        self.ctrl.send(cmd)
        self.info_label.config(text=f"🔄 스캔 시작: {session}")

    def _maybe_finalize_scan(self):
        """done 이후 tail 이미지 유입이 멈췄을 때 스캔 종료"""
        if not self._scan_done_pending:
            return
        idle_s = time.monotonic() - self._last_scan_image_ts
        if idle_s < self._scan_finalize_idle_s:
            return
        print(f"[SCAN] Finalize idle reached ({idle_s:.3f}s) -> stop_scan()")
        self._scan_done_pending = False
        self.stop_scan()

    def _on_manual_snap_saved(self, name):
        """수동 Snap 저장 완료 후 Preview 복구"""
        if not self._resume_preview_after_snap:
            return
        cfg = self._snap_preview_cfg
        self._resume_preview_after_snap = False
        self._snap_preview_cfg = None
        print(f"[SNAP] Saved: {name} -> restoring preview")
        self.root.after(150, lambda c=cfg: self._restore_preview(c, reason="snap"))

    def _snap_restore_watchdog(self, token):
        """Snap 이미지 수신 누락 시에도 Preview 복구"""
        if token != self._snap_restore_token:
            return
        if not self._resume_preview_after_snap:
            return
        cfg = self._snap_preview_cfg
        self._resume_preview_after_snap = False
        self._snap_preview_cfg = None
        print("[SNAP] Restore watchdog triggered -> restoring preview")
        self._restore_preview(cfg, reason="snap-timeout")

    def start_scan(self, params):
        """스캔 시작"""
        # ⭐ 버튼 상태 변경 (Start -> Disabled, Stop -> Normal)
        self.scan_tab.set_scan_state(True)
        self._scan_done_pending = False
        self._last_scan_image_ts = time.monotonic()

        preview_was_on = self.preview_active
        if preview_was_on:
            self._resume_preview_after_scan = True
            self._scan_preview_cfg = self._get_preview_cfg()
            print("[SCAN] Preview was ON -> pause during scan")
            w, h, fps, q = self._scan_preview_cfg
            self.toggle_preview(False, w, h, fps, q)
        else:
            self._resume_preview_after_scan = False
            self._scan_preview_cfg = None
        
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
        if preview_was_on:
            # preview 중지 명령이 먼저 적용되도록 짧게 지연
            self.info_label.config(text=f"🔄 스캔 준비 중: {session}")
            self.root.after(300, lambda c=cmd, s=session: self._send_scan_run(c, s))
        else:
            self._send_scan_run(cmd, session)
    
    def stop_scan(self):
        """스캔 중지"""
        print(f"[SCAN] Stop")
        self._scan_done_pending = False
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

        # Scan 전 Preview가 켜져 있었다면 자동 복구
        if self._resume_preview_after_scan:
            cfg = self._scan_preview_cfg
            self._resume_preview_after_scan = False
            self._scan_preview_cfg = None
            self.root.after(300, lambda c=cfg: self._restore_preview(c, reason="scan"))
    
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
        shutter, gain = self.test_tab.get_exposure_params()
        print(f"[PREVIEW] Enable={enable}, {w}x{h} @ {fps}fps, Shutter={shutter}, Gain={gain}")
        cmd = {
            "cmd": "preview",
            "enable": enable,
            "width": w,
            "height": h,
            "fps": fps,
            "quality": q,
            "shutter_speed": shutter,
            "analogue_gain": gain
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
        preview_was_on = self.preview_active
        
        # 노출 제어 파라미터 가져오기
        shutter, gain = self.test_tab.get_exposure_params()
        
        print(f"[SNAP] Capturing {w}x{h}, Shutter={shutter}, Gain={gain}")

        if preview_was_on:
            self._resume_preview_after_snap = True
            self._snap_preview_cfg = self._get_preview_cfg()
            self._snap_restore_token += 1
            token = self._snap_restore_token
            # 이미지 수신 누락 시에도 복구되도록 watchdog
            self.root.after(7000, lambda t=token: self._snap_restore_watchdog(t))
        
        # 타임스탬프
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Snap 캡처
        cmd = {
            "cmd": "snap",
            "width": w,
            "height": h,
            "quality": 95,
            "save": f"snap_{ts}.jpg",
            "shutter_speed": shutter,
            "analogue_gain": gain
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
