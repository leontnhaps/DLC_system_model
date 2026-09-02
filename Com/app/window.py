#!/usr/bin/env python3
"""
Com Client - Modular architecture with mixins
"""

import datetime
import time
from tkinter import Tk, Label, Frame, ttk
import cv2
import numpy as np

from app_config import SERVER_HOST, GUI_CTRL_PORT, GUI_IMG_PORT, SAVE_DIR
from app_state import AppState
from infra_event_bus import EventBus
from network import GuiCtrlClient, GuiImgClient, ui_q
from event_handlers import EventHandlersMixin
from pointing_handler import PointingHandlerMixin
from app_helpers import AppHelpersMixin
from ui_components import PreviewFrame, ScanTab, TestSettingsTab, PointingTab, SchedulingTab, LEDTestTab
from scan_controller import ScanController
from scheduling.proposed import ProposedScheduler, led_state_to_battery_coeff
from scheduling.round_robin import RoundRobinScheduler
from workflows.scan_workflow import ScanWorkflow
from workflows.scheduling_workflow import SchedulingWorkflow
from yolo_utils import YOLOProcessor
from led_filter import classify_from_single_roi, expand_led_roi_from_bbox, get_default_led_filter_params, led_score_to_bits
import threading

ROUNDROBIN_T_FRAME_SEC_DEFAULT = 0.0
ROUNDROBIN_T_TOTAL_SEC_DEFAULT = 0.0
ROUNDROBIN_T_SLICE_RR_FALLBACK_S = 20.0  # hidden fallback to preserve legacy RR timing if T_frame_sec is unset
ROUNDROBIN_AIM_TIMEOUT_S = 120.0
ROUNDROBIN_APPROACH_WAIT_S = 0.3         # default pre-target hold at (+1, +1)
ROUNDROBIN_FIRST_APPROACH_WAIT_S = 2.0   # first target per frame holds longer at (+1, +1)


class ComApp(EventHandlersMixin, PointingHandlerMixin, AppHelpersMixin):
    """메인 앱 - 믹스인 패턴으로 이벤트 처리 분리"""

    @property
    def laser_state(self) -> bool:
        return self.state.laser_state

    @laser_state.setter
    def laser_state(self, value: bool) -> None:
        self.state.laser_state = value

    @property
    def preview_active(self) -> bool:
        return self.state.preview_active

    @preview_active.setter
    def preview_active(self, value: bool) -> None:
        self.state.preview_active = value
    
    def __init__(self, root: Tk):
        self.root = root
        self.state = AppState()
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
        tab_scheduling = Frame(self.notebook)
        tab_led_test = Frame(self.notebook)
        
        self.notebook.add(tab_scan, text="Scan")
        self.notebook.add(tab_test, text="Test & Settings")
        self.notebook.add(tab_pointing, text="Pointing")
        self.notebook.add(tab_scheduling, text="Scheduling")
        self.notebook.add(tab_led_test, text="LED Test")
        self._tab_index_pointing = 2
        
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
            'set_preview_crosshair': self.set_preview_crosshair,
            'set_ir_cut': self.set_ir_cut,
            'snap_capture': self.snap_capture
        }
        self.test_tab = TestSettingsTab(tab_test, test_callbacks)
        
        pointing_callbacks = {
            'pointing_choose_csv': self.pointing_choose_csv,
            'pointing_compute': self.pointing_compute,
            'set_pointing_mode': self.set_pointing_mode,
            'move_to_target': self.move_to_target,
            'start_aiming': self.start_aiming,
            'stop_aiming': self.stop_aiming
        }
        self.pointing_tab = PointingTab(tab_pointing, pointing_callbacks)

        scheduling_callbacks = {
            "start_roundrobin": self.start_roundrobin,
            "start_proposed": self.start_proposed_scheduling,
            "stop_scheduling": self.stop_scheduling,
        }
        self.scheduling_tab = SchedulingTab(tab_scheduling, scheduling_callbacks)

        led_test_callbacks = {
            "start_led_test": self.start_led_test,
            "stop_led_test": self.stop_led_test,
        }
        self.led_test_tab = LEDTestTab(
            tab_led_test,
            led_test_callbacks,
            initial_params=get_default_led_filter_params(),
        )
        
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
        self._scan_finished_event = threading.Event()
        self._scan_finished_event.set()
        self._last_scan_result = None
        
        # Scheduling 상태
        self._scheduling_active = False
        self._scheduling_thread = None
        self._scheduling_stop_event = threading.Event()
        self._scheduling_led_latest = {}
        self._scheduling_led_history = []
        self._track_led_roi = {}  # {track_id: (x,y,w,h)}
        self._track_led_roi_source_size = {}  # {track_id: (W,H)}
        self._track_final_led_state = {}  # {track_id: {"pred": str, "score": {"R":int,"G":int,"B":int}}}
        self._track_phase3_response = {}  # {track_id: {"mean": float, "core": float, "max": float}}
        self._scheduling_frame_debug = {}
        self._led_test_active = False
        self._led_test_thread = None
        self._led_test_stop_event = threading.Event()
        self._led_test_cached_roi = None
        self._led_test_cached_bbox = None
        self._led_test_cached_all_bboxes = []
        
        # Blocking snap wait state (Scheduling probe 등에서 사용)
        self._blocking_snap_lock = threading.Lock()
        self._blocking_snap_event = threading.Event()
        self._blocking_snap_expected_name = None
        self._blocking_snap_data = None
        
        # LED filter params (shared with scan/pointing/scheduling)
        self.led_filter_params = get_default_led_filter_params()
        
        # Scan Controller
        self.scan_ctrl = ScanController(
            SAVE_DIR,
            self.yolo_processor,
            led_filter_params=self.led_filter_params,
        )
        self.scan_workflow = ScanWorkflow(self.scan_ctrl)
        self.scheduling_workflow = SchedulingWorkflow()
        if hasattr(self.scheduling_workflow, "set_context"):
            self.scheduling_workflow.set_context(app=self)
        self._scheduling_mode_key = "round_robin"
        self._scheduling_mode_label = "RoundRobin"

        # Pointing related initializations (from PointingHandlerMixin)
        self._pointing_gains = {}
        self._pointing_img_event = threading.Event()
        self.pointing_mode = "adaptive"
        
        # Frame count (for preview)
        self.frame_count = 0

        # Event bus (wrap existing ui_q for backward compatibility)
        self.bus = EventBus(ui_q)
        
        # Network Clients
        self.ctrl = GuiCtrlClient(SERVER_HOST, GUI_CTRL_PORT, bus=self.bus)
        self.ctrl.start()
        
        self.img = GuiImgClient(SERVER_HOST, GUI_IMG_PORT, SAVE_DIR, bus=self.bus)
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

    @staticmethod
    def _normalize_led_bits_for_preview(value):
        text = str(value or "").strip()
        if 1 <= len(text) <= 3 and all(ch in "01" for ch in text):
            return text.zfill(3)
        return None

    def _get_preview_led_bits(self, track_id=None, fallback=None):
        bits = None
        track_key = None
        if track_id is not None:
            try:
                track_key = int(track_id)
            except Exception:
                track_key = track_id
            final_states = getattr(self, "_track_final_led_state", {}) or {}
            state_info = final_states.get(track_key)
            if state_info is None and track_key is not track_id:
                state_info = final_states.get(track_id)
            if isinstance(state_info, dict):
                bits = self._normalize_led_bits_for_preview(state_info.get("bits"))

        if bits is None:
            bits = self._normalize_led_bits_for_preview(fallback)

        if bits is None and track_key is not None:
            latest_states = getattr(self, "_scheduling_led_latest", {}) or {}
            bits = self._normalize_led_bits_for_preview(latest_states.get(track_key))
            if bits is None and track_key is not track_id:
                bits = self._normalize_led_bits_for_preview(latest_states.get(track_id))

        return bits or "-"

    def _set_preview_overlay(self, current_id=None, phase="Idle", dwell_elapsed=None, dwell_total=None, led_state=None):
        """프리뷰 오버레이에 진행 시간과 CSV/스케줄링 LED bit 표시"""
        cid = "-" if current_id is None else str(current_id)
        bits = self._get_preview_led_bits(current_id, fallback=led_state)
        if dwell_elapsed is not None and dwell_total is not None and dwell_total > 0:
            time_text = f"Slice: {dwell_elapsed:.1f}/{dwell_total:.1f}s"
        else:
            time_text = "Slice: -"
        text = f"{time_text} | ID: {cid} | BIT: {bits}"
        if hasattr(self, "preview_frame") and hasattr(self.preview_frame, "set_overlay_text"):
            self.root.after(0, lambda t=text: self.preview_frame.set_overlay_text(t))
        if hasattr(self, "preview_frame") and hasattr(self.preview_frame, "set_overlay_roi"):
            self.root.after(
                0,
                lambda: self.preview_frame.set_overlay_roi(None, None),
            )

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
        if self._led_test_active:
            self.info_label.config(text="⚠️ LED Test 실행 중에는 Scan을 시작할 수 없습니다.")
            return
        # ⭐ 버튼 상태 변경 (Start -> Disabled, Stop -> Normal)
        self.scan_tab.set_scan_state(True)
        self._scan_done_pending = False
        self._last_scan_image_ts = time.monotonic()
        self._scan_finished_event.clear()
        self._last_scan_result = None

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
        
        # LED 인식은 Normal(가시광선) 모드 기반으로 처리
        self.set_ir_cut("night")
        time.sleep(0.05)
        
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
        self._last_scan_result = result
        
        # UI 업데이트
        if result:
            self.info_label.config(text=f"⏹️ 스캔 중지: {result['done']}/{result['total']}")
            self.scan_tab.set_scan_state(False)
            
            # Pointing 자동 실행
            csv_path = result.get('csv_path_abs')
            if csv_path:
                if self._scheduling_active and self._scheduling_stop_event.is_set():
                    print("[ComApp] Scheduling stop requested -> skip auto-pointing compute")
                else:
                    print(f"[ComApp] Auto-computing pointing for: {csv_path}")
                    # Pointing 탭으로 전환 (선택 사항)
                    if not self._scheduling_active:
                        self.notebook.select(self._tab_index_pointing)
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

        self._scan_finished_event.set()
        return result
    
    # ========== Scheduling Callbacks ==========
    def _call_on_ui_thread(self, fn, timeout=10.0):
        """Worker thread에서 UI thread 함수 안전 호출"""
        done_evt = threading.Event()
        holder = {}

        def _runner():
            try:
                holder["result"] = fn()
            except Exception as exc:
                holder["error"] = exc
            finally:
                done_evt.set()

        self.root.after(0, _runner)
        if not done_evt.wait(timeout):
            raise TimeoutError("UI thread call timeout")
        if "error" in holder:
            raise holder["error"]
        return holder.get("result")
    
    def _notify_blocking_snap_saved(self, name, data):
        """
        saved 이벤트에서 blocking-snap 대기자에게 데이터 전달.
        반환값은 소비 여부가 아니라 '대기자에게 전달했는지'만 의미.
        """
        with self._blocking_snap_lock:
            expected = self._blocking_snap_expected_name
            if not expected:
                return False
            if name != expected:
                return False
            self._blocking_snap_data = data
            self._blocking_snap_event.set()
            return True
    
    def _blocking_snap_and_wait(self, save_name, timeout=10.0, shutter_speed=None, analogue_gain=None):
        """Scheduling 등에서 사용할 blocking snap helper (thread-safe)."""
        if not save_name.lower().endswith(".jpg"):
            save_name = f"{save_name}.jpg"

        with self._blocking_snap_lock:
            self._blocking_snap_expected_name = save_name
            self._blocking_snap_data = None
            self._blocking_snap_event.clear()

        w = self.scan_tab.width.get()
        h = self.scan_tab.height.get()
        q = self.scan_tab.quality.get()
        cmd = {
            "cmd": "snap",
            "width": int(w),
            "height": int(h),
            "quality": int(q),
            "save": save_name,
        }
        if shutter_speed is not None:
            cmd["shutter_speed"] = int(shutter_speed)
        if analogue_gain is not None:
            cmd["analogue_gain"] = float(analogue_gain)
        self.ctrl.send(cmd)

        deadline = time.monotonic() + float(timeout)
        try:
            while True:
                remain = deadline - time.monotonic()
                if remain <= 0:
                    print(f"[Scheduling] Snap timeout: {save_name}")
                    return None
                if self._scheduling_stop_event.is_set() or self._led_test_stop_event.is_set():
                    return None
                if self._blocking_snap_event.wait(timeout=min(0.1, remain)):
                    with self._blocking_snap_lock:
                        data = self._blocking_snap_data
                    if not data:
                        return None
                    arr = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    return img
        finally:
            with self._blocking_snap_lock:
                self._blocking_snap_expected_name = None
                self._blocking_snap_data = None
                self._blocking_snap_event.clear()

    def _set_led_test_ui_state(self, is_running):
        def _update():
            if hasattr(self, "led_test_tab"):
                self.led_test_tab.set_running_state(is_running)
        self.root.after(0, _update)

    def _set_led_test_status(self, text, fg="#333"):
        def _update():
            if hasattr(self, "led_test_tab"):
                self.led_test_tab.update_status(text, fg=fg)
        self.root.after(0, _update)

    def _update_led_test_result(
        self,
        raw_score=None,
        bits="000",
        threshold=None,
        roi=None,
        legacy_pred="NONE",
        preview_img=None,
        status_text=None,
        status_fg="#333",
    ):
        def _update():
            if not hasattr(self, "led_test_tab"):
                return
            if status_text is not None:
                self.led_test_tab.update_status(status_text, fg=status_fg)
            self.led_test_tab.update_result(
                raw_score=raw_score,
                bits=bits,
                threshold=threshold,
                roi=roi,
                legacy_pred=legacy_pred,
            )
            if preview_img is not None:
                self.led_test_tab.show_preview(preview_img)
        self.root.after(0, _update)

    def _build_led_test_preview_image(self, img_bgr, bbox=None, roi=None, all_bboxes=None, bits="000", score=None):
        if img_bgr is None or img_bgr.size == 0:
            return np.zeros((300, 420, 3), dtype=np.uint8)

        vis = img_bgr.copy()
        h_img, w_img = vis.shape[:2]
        center_x = w_img // 2
        center_y = h_img // 2
        cv2.drawMarker(vis, (center_x, center_y), (255, 255, 255), cv2.MARKER_CROSS, 36, 2)

        for candidate in list(all_bboxes or []):
            try:
                x, y, w, h = [int(round(float(v))) for v in candidate]
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), 2)
            except Exception:
                continue

        if bbox is not None:
            try:
                x, y, w, h = [int(round(float(v))) for v in bbox]
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
            except Exception:
                pass

        if roi is not None:
            try:
                rx, ry, rw, rh = [int(round(float(v))) for v in roi]
                cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 3)
            except Exception:
                pass

        # ROI가 잡힌 뒤에는 ROI 주변을 crop해서 확대된 preview로 표시한다.
        preview = vis
        if roi is not None:
            try:
                rx, ry, rw, rh = [int(round(float(v))) for v in roi]
                cx = rx + (rw // 2)
                cy = ry + (rh // 2)
                half_w = max(140, int(round(rw * 2.2)))
                half_h = max(110, int(round(rh * 2.2)))
                x1 = max(0, cx - half_w)
                x2 = min(w_img, cx + half_w)
                y1 = max(0, cy - half_h)
                y2 = min(h_img, cy + half_h)
                if x2 > x1 and y2 > y1:
                    preview = vis[y1:y2, x1:x2].copy()
            except Exception:
                preview = vis

        return preview

    def _capture_led_test_pair(self, led_value=255, led_settle_s=0.2):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        snap_on = f"led_test_on_{ts}.jpg"
        snap_off = f"led_test_off_{ts}.jpg"

        try:
            self._call_on_ui_thread(lambda: self.set_ir_cut("night"), timeout=3.0)
        except Exception:
            pass

        try:
            self.ctrl.send({"cmd": "laser", "value": 0})
            self.laser_state = False
        except Exception:
            pass

        self._call_on_ui_thread(lambda v=int(led_value): self.set_led(v), timeout=3.0)
        time.sleep(max(0.05, float(led_settle_s)))
        img_on = self._blocking_snap_and_wait(
            snap_on,
            timeout=10.0,
            shutter_speed=10000,
            analogue_gain=1.0,
        )

        self._call_on_ui_thread(lambda: self.set_led(0), timeout=3.0)
        time.sleep(max(0.05, float(led_settle_s)))
        img_off = self._blocking_snap_and_wait(
            snap_off,
            timeout=10.0,
            shutter_speed=10000,
            analogue_gain=1.0,
        )
        return img_on, img_off

    def _capture_scheduling_led_probe_pair(self, track_id, led_value=255, led_settle_s=0.2, timeout_s=10.0):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        snap_on = f"sched_led_on_id{track_id}_{ts}.jpg"
        snap_off = f"sched_led_off_id{track_id}_{ts}.jpg"

        self._call_on_ui_thread(lambda v=int(led_value): self.set_led(v), timeout=3.0)
        time.sleep(max(0.05, float(led_settle_s)))
        img_on = self._blocking_snap_and_wait(
            snap_on,
            timeout=max(0.2, float(timeout_s)),
            shutter_speed=10000,
            analogue_gain=1.0,
        )

        self._call_on_ui_thread(lambda: self.set_led(0), timeout=3.0)
        time.sleep(max(0.05, float(led_settle_s)))
        img_off = self._blocking_snap_and_wait(
            snap_off,
            timeout=max(0.2, float(timeout_s)),
            shutter_speed=10000,
            analogue_gain=1.0,
        )
        return img_on, img_off

    def _capture_led_test_single(self):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        snap_name = f"led_test_single_{ts}.jpg"

        try:
            self._call_on_ui_thread(lambda: self.set_ir_cut("night"), timeout=3.0)
        except Exception:
            pass

        try:
            self.ctrl.send({"cmd": "laser", "value": 0})
            self.laser_state = False
        except Exception:
            pass

        try:
            self._call_on_ui_thread(lambda: self.set_led(0), timeout=3.0)
        except Exception:
            pass

        return self._blocking_snap_and_wait(
            snap_name,
            timeout=10.0,
            shutter_speed=10000,
            analogue_gain=1.0,
        )

    def start_led_test(self):
        if self._led_test_active:
            self._set_led_test_status("⚠️ LED Test already running", fg="orange")
            return False
        if self.scan_ctrl.is_active():
            self._set_led_test_status("⚠️ Scan 실행 중에는 LED Test를 시작할 수 없습니다.", fg="orange")
            return False
        if self._scheduling_active:
            self._set_led_test_status("⚠️ Scheduling 실행 중에는 LED Test를 시작할 수 없습니다.", fg="orange")
            return False
        if getattr(self, "_aiming_active", False):
            self._set_led_test_status("⚠️ Pointing 실행 중에는 LED Test를 시작할 수 없습니다.", fg="orange")
            return False

        self._led_test_active = True
        self._led_test_stop_event.clear()
        self._led_test_cached_roi = None
        self._led_test_cached_bbox = None
        self._led_test_cached_all_bboxes = []
        self._set_led_test_ui_state(True)
        self._set_led_test_status("🔍 LED Test 시작...", fg="blue")
        self._led_test_thread = threading.Thread(target=self._led_test_worker, daemon=True)
        self._led_test_thread.start()
        return True

    def stop_led_test(self):
        self._led_test_stop_event.set()
        self._set_led_test_status("⛔ LED Test 중지 요청...", fg="red")
        self._led_test_cached_roi = None
        self._led_test_cached_bbox = None
        self._led_test_cached_all_bboxes = []
        try:
            self._call_on_ui_thread(lambda: self.set_led(0), timeout=3.0)
        except Exception:
            pass
        if not self._led_test_active:
            self._set_led_test_ui_state(False)

    def _led_test_worker(self):
        final_message = "✅ LED Test 종료"
        final_color = "green"
        try:
            while not self._led_test_stop_event.is_set():
                params = dict(self._call_on_ui_thread(lambda: self.led_test_tab.get_filter_params(), timeout=2.0))
                led_settle_s = float(self._call_on_ui_thread(lambda: self.scan_tab.led_settle.get(), timeout=2.0))

                bbox = self._led_test_cached_bbox
                all_bboxes = list(self._led_test_cached_all_bboxes or [])
                roi = self._led_test_cached_roi
                detected_this_cycle = False
                img_off = None

                if roi is None:
                    img_on, img_off = self._capture_led_test_pair(led_value=255, led_settle_s=led_settle_s)
                    if self._led_test_stop_event.is_set():
                        break
                    if img_on is None or img_off is None:
                        self._update_led_test_result(
                            raw_score={"R": 0, "G": 0, "B": 0},
                            bits="000",
                            threshold=params.get("min_pixels", 0),
                            roi=None,
                            legacy_pred="NONE",
                            preview_img=np.zeros((300, 420, 3), dtype=np.uint8),
                            status_text="⚠️ LED Test snap 실패, 재시도 중...",
                            status_fg="orange",
                        )
                        time.sleep(0.5)
                        continue
                    obj_cx, obj_cy, bbox, all_bboxes = self._find_object_center(img_on, img_off, selection_roi_box=None)
                    _ = (obj_cx, obj_cy)
                    led_info = dict(getattr(self, "_last_object_led_info", {}) or {})
                    roi = led_info.get("roi")
                    if roi is None and bbox is not None:
                        roi = expand_led_roi_from_bbox(bbox, img_off.shape, top_ratio=1.0 / 3.0)
                    if bbox is not None and roi is not None:
                        self._led_test_cached_bbox = tuple(int(v) for v in bbox)
                        self._led_test_cached_all_bboxes = [tuple(int(v) for v in bb) for bb in (all_bboxes or [])]
                        self._led_test_cached_roi = tuple(int(v) for v in roi)
                        detected_this_cycle = True
                else:
                    img_off = self._capture_led_test_single()
                    if self._led_test_stop_event.is_set():
                        break
                    if img_off is None:
                        self._update_led_test_result(
                            raw_score={"R": 0, "G": 0, "B": 0},
                            bits="000",
                            threshold=params.get("min_pixels", 0),
                            roi=roi,
                            legacy_pred="NONE",
                            preview_img=np.zeros((300, 420, 3), dtype=np.uint8),
                            status_text="⚠️ LED Test single snap 실패, 재시도 중...",
                            status_fg="orange",
                        )
                        time.sleep(0.5)
                        continue
                    roi = tuple(int(v) for v in roi)
                    if bbox is not None:
                        bbox = tuple(int(v) for v in bbox)
                    all_bboxes = [tuple(int(v) for v in bb) for bb in (all_bboxes or [])]

                legacy_pred = "NONE"
                score = {"R": 0, "G": 0, "B": 0}
                bits = "000"
                preview_img = self._build_led_test_preview_image(
                    img_off,
                    bbox=bbox,
                    roi=roi,
                    all_bboxes=all_bboxes,
                    bits=bits,
                    score=score,
                )

                if bbox is not None and roi is not None:
                    legacy_pred, score, roi_used = classify_from_single_roi(img_off, roi, params=params)
                    if roi_used is not None:
                        roi = roi_used
                        self._led_test_cached_roi = tuple(int(v) for v in roi_used)
                    bit_result = led_score_to_bits(score, threshold=params.get("min_pixels", 0))
                    bits = bit_result["bits"]
                    preview_img = self._build_led_test_preview_image(
                        img_off,
                        bbox=bbox,
                        roi=roi,
                        all_bboxes=all_bboxes,
                        bits=bits,
                        score=score,
                    )
                    if detected_this_cycle:
                        status_text = f"✅ Target detected, ROI locked | legacy={legacy_pred} | bits={bits}"
                    else:
                        status_text = f"✅ ROI reused | legacy={legacy_pred} | bits={bits}"
                    status_fg = "green"
                else:
                    status_text = "⚠️ 타깃 검출 실패"
                    status_fg = "orange"

                self._update_led_test_result(
                    raw_score=score,
                    bits=bits,
                    threshold=params.get("min_pixels", 0),
                    roi=roi,
                    legacy_pred=legacy_pred,
                    preview_img=preview_img,
                    status_text=status_text,
                    status_fg=status_fg,
                )

                for _ in range(10):
                    if self._led_test_stop_event.is_set():
                        break
                    time.sleep(0.1)
        except Exception as e:
            final_message = f"❌ LED Test failed: {e}"
            final_color = "red"
            print(f"[LED Test] Worker failed: {e}")
        finally:
            self._led_test_active = False
            self._led_test_thread = None
            self._led_test_stop_event.clear()
            self._led_test_cached_roi = None
            self._led_test_cached_bbox = None
            self._led_test_cached_all_bboxes = []
            try:
                self._call_on_ui_thread(lambda: self.set_led(0), timeout=3.0)
            except Exception:
                pass
            def _finalize_led_test_ui(msg=final_message, fg=final_color):
                if hasattr(self, "led_test_tab"):
                    self.led_test_tab.set_running_state(False)
                    self.led_test_tab.update_status(msg, fg=fg)
            self.root.after(0, _finalize_led_test_ui)

    def _probe_led_state_for_track(self, track_id, probe_interval_s=2.0, timeout_s=10.0, return_bits=False, refresh_roi=False):
        """
        Scheduling shoot loop 중 K초 주기 LED 상태 프로브.
        기본은 저장된 ROI에서 단일 프레임으로 LED 상태를 판정한다.
        Proposed sampling에서는 LED ON/OFF pair로 객체를 다시 찾고 ROI를 재생성할 수 있다.
        카메라 모드는 shoot loop의 현재 상태를 그대로 유지한다.
        """
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        snap_name = f"sched_led_single_id{track_id}_{ts}.jpg"

        preview_was_on = bool(self._call_on_ui_thread(lambda: self.preview_active, timeout=2.0))
        pred = "NONE"
        bits = "000"
        score = {"R": 0, "G": 0, "B": 0}
        roi = None
        try:
            img = self._blocking_snap_and_wait(
                snap_name,
                timeout=max(0.2, float(timeout_s)),
                shutter_speed=10000,
                analogue_gain=None,
            ) if not refresh_roi else None

            scheduling_led_params = dict(getattr(self, "led_filter_params", None) or get_default_led_filter_params())
            scheduling_led_params["min_pixels"] = 50

            if refresh_roi:
                try:
                    led_settle_s = float(getattr(self.scan_tab, "led_settle", None).get())
                except Exception:
                    led_settle_s = 0.2
                pair_timeout = max(0.2, min(float(timeout_s), 10.0))
                img_on, img_off = self._capture_scheduling_led_probe_pair(
                    track_id,
                    led_value=255,
                    led_settle_s=led_settle_s,
                    timeout_s=pair_timeout,
                )
                if img_on is None or img_off is None:
                    return "NONE"
                _obj_cx, _obj_cy, bbox, _all_bboxes = self._find_object_center(
                    img_on,
                    img_off,
                    selection_roi_box=None,
                )
                if bbox is None:
                    print(f"[Scheduling] LED probe skipped (ID {track_id}): no detected object for ROI refresh")
                    return "NONE"
                led_roi_seed = expand_led_roi_from_bbox(
                    bbox,
                    img_off.shape,
                    top_ratio=1.0 / 3.0,
                )
                if led_roi_seed is not None and hasattr(self, "_expand_loaded_led_roi_x"):
                    led_roi_seed = self._expand_loaded_led_roi_x(
                        led_roi_seed,
                        (int(img_off.shape[1]), int(img_off.shape[0])),
                    )
                pred, score, roi_used = classify_from_single_roi(
                    img_off,
                    led_roi_seed,
                    params=scheduling_led_params,
                )
                bit_result = led_score_to_bits(score, threshold=scheduling_led_params.get("min_pixels", 0))
                bits = str(bit_result["bits"])
                if max(int(score["R"]), int(score["G"]), int(score["B"])) < int(scheduling_led_params["min_pixels"]):
                    pred = "R"
                if roi_used is not None:
                    self._track_led_roi[track_id] = tuple(int(v) for v in roi_used)
                    self._track_led_roi_source_size[track_id] = (int(img_off.shape[1]), int(img_off.shape[0]))
                roi = roi_used
            else:
                roi = self._track_led_roi.get(track_id)
                if roi is None:
                    print(f"[Scheduling] LED probe skipped (ID {track_id}): no stored ROI")
                    return "NONE"
                if img is None:
                    return "NONE"

                pred, score, roi_used = classify_from_single_roi(
                    img,
                    roi,
                    params=scheduling_led_params,
                )
                bit_result = led_score_to_bits(score, threshold=scheduling_led_params.get("min_pixels", 0))
                bits = str(bit_result["bits"])
                if max(int(score["R"]), int(score["G"]), int(score["B"])) < int(scheduling_led_params["min_pixels"]):
                    pred = "R"
                if roi_used is not None:
                    self._track_led_roi[track_id] = tuple(int(v) for v in roi_used)
                    self._track_led_roi_source_size[track_id] = (int(img.shape[1]), int(img.shape[0]))
                roi = roi_used

            self._scheduling_led_latest[track_id] = pred
            self._scheduling_led_history.append({
                "ts": ts,
                "track_id": int(track_id),
                "pred": pred,
                "bits": bits,
                "r": int(score["R"]),
                "g": int(score["G"]),
                "b": int(score["B"]),
                "roi": tuple(int(v) for v in roi) if roi is not None else None,
                "mode": "pair_refresh_roi" if refresh_roi else "single_roi",
                "probe_interval_s": float(probe_interval_s),
            })
            return bits if return_bits else pred
        except Exception as e:
            print(f"[Scheduling] LED probe failed (ID {track_id}): {e}")
            return "NONE"
        finally:
            if preview_was_on:
                try:
                    self._call_on_ui_thread(
                        lambda: self.toggle_preview(True, *self._get_preview_cfg()),
                        timeout=4.0,
                    )
                except Exception:
                    pass

    def _set_scheduling_ui_state(self, is_running):
        def _update():
            if hasattr(self, "scheduling_tab"):
                self.scheduling_tab.set_running_state(is_running)
        self.root.after(0, _update)

    def _get_scheduling_backend(self):
        """Return scheduling workflow backend when available."""
        return self.scheduling_workflow if hasattr(self, "scheduling_workflow") else None

    def _configure_scheduling_backend(self, mode_key):
        backend = self._get_scheduling_backend()
        mode_key = str(mode_key or "round_robin").strip().lower()
        if mode_key == "proposed":
            scheduler = ProposedScheduler()
            label = "Proposed"
        else:
            scheduler = RoundRobinScheduler()
            label = "RoundRobin"
            mode_key = "round_robin"

        if backend and hasattr(backend, "set_scheduler"):
            backend.set_scheduler(scheduler)
        self._scheduling_mode_key = mode_key
        self._scheduling_mode_label = label
        return label

    def _order_scheduling_target_ids(self, targets):
        """Scheduling target order from the active backend."""
        backend = self._get_scheduling_backend()
        if backend and hasattr(backend, "order_target_ids"):
            return backend.order_target_ids(targets)
        return [track_id for track_id, _ in sorted(
            (targets or {}).items(),
            key=lambda item: (-float(item[1][0]), int(item[0])),
        )]

    def _resolve_roundrobin_timing(self, n_targets, frame_sec=None, total_sec=None):
        """Resolve canonical RR timing in frame-based form."""
        N_targets = max(1, int(n_targets))

        try:
            T_frame_sec = float(frame_sec)
        except Exception:
            T_frame_sec = 0.0
        if T_frame_sec <= 0.0:
            T_frame_sec = 0.0

        try:
            T_total_sec = float(total_sec)
        except Exception:
            T_total_sec = 0.0
        if T_total_sec <= 0.0:
            T_total_sec = 0.0

        if T_frame_sec > 0.0:
            t_slice_rr = T_frame_sec / float(N_targets)
        else:
            t_slice_rr = float(max(0.2, ROUNDROBIN_T_SLICE_RR_FALLBACK_S))
            T_frame_sec = t_slice_rr * float(N_targets)

        K_frames = None
        requested_total_sec = float(T_total_sec)
        if T_total_sec > 0.0 and T_frame_sec > 0.0:
            K_frames = max(1, int(round(T_total_sec / T_frame_sec)))
            T_total_sec = float(T_frame_sec) * float(K_frames)

        return {
            "T_frame_sec": float(T_frame_sec),
            "T_total_sec": float(T_total_sec),
            "T_total_sec_requested": float(requested_total_sec),
            "K_frames": K_frames,
            "N_targets": N_targets,
            "t_slice_rr": float(t_slice_rr),
            "dwell_s": float(t_slice_rr),  # compatibility alias for existing helper/overlay naming
        }

    @staticmethod
    def _format_clock_label(seconds):
        if seconds is None:
            return "-"
        try:
            seconds = max(0.0, float(seconds))
        except Exception:
            return "-"
        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def _format_slice_label(seconds):
        if seconds is None:
            return "-"
        try:
            seconds = max(0.0, float(seconds))
        except Exception:
            return "-"
        return f"{seconds:.1f}s"

    def _wait_until_scheduling_deadline(self, deadline_s, poll_s=0.05):
        """Wait until deadline while allowing prompt scheduling stop."""
        while not self._scheduling_stop_event.is_set():
            remain = float(deadline_s) - time.monotonic()
            if remain <= 0.0:
                return True
            time.sleep(min(max(0.01, poll_s), remain))
        return False

    def _render_scheduling_progress(self, text, fg="#555"):
        def _update():
            if hasattr(self, "scheduling_tab"):
                self.scheduling_tab.update_progress_text(text, fg=fg)

        self.root.after(0, _update)

    def _set_scheduling_frame_debug(self, **kwargs):
        debug = dict(getattr(self, "_scheduling_frame_debug", {}) or {})
        debug.update(kwargs)
        self._scheduling_frame_debug = debug

    def _clear_scheduling_frame_debug(self):
        self._scheduling_frame_debug = {}

    def _reset_scheduling_progress(self):
        self._clear_scheduling_frame_debug()
        def _update():
            if hasattr(self, "scheduling_tab"):
                self.scheduling_tab.reset_progress_text()

        self.root.after(0, _update)

    def _update_scheduling_progress(
        self,
        scheduling_elapsed_s=None,
        shoot_elapsed_s=None,
        shoot_total_s=None,
        phase="Idle",
        current_id=None,
        loop_count=None,
        frame_elapsed_s=None,
        frame_total_s=None,
        slice_index=None,
        slice_total=None,
        current_elapsed_s=None,
        current_total_s=None,
        completed_frames=None,
        total_frames=None,
        fg="#555",
    ):
        debug = dict(getattr(self, "_scheduling_frame_debug", {}) or {})
        mode_label = str(debug.get("mode_label") or getattr(self, "_scheduling_mode_label", "-"))
        mode_progress_label = f"{mode_label} 진행"
        frame_allocations = dict(debug.get("frame_allocations") or {})
        execution_order = list(debug.get("execution_order") or frame_allocations.keys())
        fixed_target_coeffs = dict(debug.get("fixed_target_coeffs") or {})
        battery_state_prev = dict(debug.get("battery_state_prev") or {})
        battery_coeff_prev = dict(debug.get("battery_coeff_prev") or {})
        frame_scores = dict(debug.get("frame_scores") or {})

        overall_text = f"전체 경과: {self._format_clock_label(scheduling_elapsed_s)}"
        mode_text = f"모드: {mode_label}"

        if shoot_elapsed_s is None and not (shoot_total_s is not None and shoot_total_s > 0.0):
            shoot_text = f"{mode_progress_label}: 준비 중"
        elif shoot_total_s is not None and shoot_total_s > 0.0:
            remaining_s = max(0.0, float(shoot_total_s) - float(shoot_elapsed_s or 0.0))
            shoot_text = (
                f"{mode_progress_label}: {self._format_clock_label(shoot_elapsed_s)} / "
                f"{self._format_clock_label(shoot_total_s)} "
                f"(남은 {self._format_clock_label(remaining_s)})"
            )
        else:
            shoot_text = f"{mode_progress_label}: {self._format_clock_label(shoot_elapsed_s)} / manual"

        if loop_count is not None:
            frame_line = f"현재 Frame: {int(loop_count)}"
            if total_frames is not None:
                frame_line += f" / {int(total_frames)}"
        elif completed_frames is not None:
            frame_line = f"현재 Frame: {int(completed_frames)}"
            if total_frames is not None:
                frame_line += f" / {int(total_frames)}"
        else:
            frame_line = "현재 Frame: -"

        target_line = f"현재 타깃: ID {int(current_id)}" if current_id is not None else "현재 타깃: -"

        detail_parts = []
        if phase:
            detail_parts.append(str(phase))
        if frame_total_s is not None and frame_total_s > 0.0:
            frame_idx_text = str(int(loop_count)) if loop_count is not None else "-"
            frame_total_text = str(int(total_frames)) if total_frames is not None else "manual"
            detail_parts.append(
                f"Frame {frame_idx_text}/{frame_total_text} "
                f"{self._format_clock_label(frame_elapsed_s)}/"
                f"{self._format_clock_label(frame_total_s)}"
            )
        elif loop_count is not None:
            detail_parts.append(f"Frame {int(loop_count)}")
        if slice_index is not None and slice_total is not None:
            detail_parts.append(f"Slice {int(slice_index)}/{int(slice_total)}")
        if current_id is not None:
            detail_parts.append(f"ID {current_id}")
        if current_total_s is not None and current_total_s > 0.0:
            detail_parts.append(
                f"Slice {self._format_slice_label(current_elapsed_s or 0.0)}/"
                f"{self._format_slice_label(current_total_s)}"
            )
        if completed_frames is not None and frame_total_s is None:
            frame_total_text = str(int(total_frames)) if total_frames is not None else "manual"
            detail_parts.append(f"Frame {int(completed_frames)}/{frame_total_text}")

        detail_text = " | ".join(detail_parts) if detail_parts else "-"
        allocation_lines = []
        if frame_allocations:
            allocation_lines.append("현재 Frame 할당:")
            for track_id in execution_order:
                try:
                    alloc_s = float(frame_allocations.get(track_id, 0.0))
                except Exception:
                    alloc_s = 0.0
                line = f"Track {int(track_id)} -> {alloc_s:.1f} s"
                extras = []
                coeff_c = fixed_target_coeffs.get(track_id)
                coeff_b = battery_coeff_prev.get(track_id)
                state_b = battery_state_prev.get(track_id)
                score_v = frame_scores.get(track_id)
                if coeff_c is not None:
                    extras.append(f"C={float(coeff_c):.3f}")
                if coeff_b is not None or state_b is not None:
                    coeff_txt = "-" if coeff_b is None else f"{float(coeff_b):.2f}"
                    state_txt = str(state_b or "-")
                    extras.append(f"Bprev={coeff_txt}({state_txt})")
                if score_v is not None:
                    extras.append(f"Score={float(score_v):.3f}")
                if extras:
                    line += " | " + " | ".join(extras)
                allocation_lines.append(line)

        progress_lines = [
            mode_text,
            overall_text,
            shoot_text,
            frame_line,
            target_line,
            f"현재 작업: {detail_text}",
        ]
        progress_lines.extend(allocation_lines)
        self._render_scheduling_progress(
            "\n".join(progress_lines),
            fg=fg,
        )

    def _set_scheduling_status(self, text, fg="#333"):
        def _update():
            if hasattr(self, "scheduling_tab"):
                self.scheduling_tab.update_status(text, fg=fg)
            self.info_label.config(text=text)
        self.root.after(0, _update)

    def _finalize_scheduling_ui(self, message, fg="#333"):
        self._scheduling_active = False
        self._scheduling_thread = None
        self._scheduling_stop_event.clear()
        self._clear_scheduling_frame_debug()
        if hasattr(self, "scheduling_tab"):
            self.scheduling_tab.set_running_state(False)
            self.scheduling_tab.update_status(message, fg=fg)
        self.info_label.config(text=message)
        self._set_preview_overlay(current_id=None, phase="Idle")

    def _start_scheduling(self, mode_key="round_robin"):
        """Start scheduling using the requested backend."""
        backend = self._get_scheduling_backend()
        if backend and hasattr(backend, "set_context"):
            backend.set_context(app=self)
        mode_label = self._configure_scheduling_backend(mode_key)

        if self._scheduling_active:
            self._set_scheduling_status("⚠️ Scheduling already running", fg="orange")
            return False
        if self.scan_ctrl.is_active():
            self._set_scheduling_status("⚠️ Scan already running", fg="orange")
            return False
        if getattr(self, "_aiming_active", False):
            self._set_scheduling_status("⚠️ Pointing is running. Stop aiming first.", fg="orange")
            return False
        if self._led_test_active:
            self._set_scheduling_status("⚠️ LED Test is running. Stop LED Test first.", fg="orange")
            return False

        self._scheduling_active = True
        self._scheduling_stop_event.clear()
        self._set_scheduling_ui_state(True)
        self._reset_scheduling_progress()
        self._set_scheduling_status(f"🔁 {mode_label} 시작...", fg="blue")

        self._scheduling_thread = threading.Thread(target=self._roundrobin_worker, daemon=True)
        self._scheduling_thread.start()
        return True

    def start_roundrobin(self):
        """RoundRobin 스케줄 시작"""
        return self._start_scheduling("round_robin")

    def start_proposed_scheduling(self):
        """Proposed scheduling 시작"""
        return self._start_scheduling("proposed")

    def stop_scheduling(self):
        """현재 실행 중인 Scheduling 알고리즘 중지"""
        self._scheduling_stop_event.set()
        self._set_scheduling_status("⛔ Scheduling 중지 요청...", fg="red")

        if self.scan_ctrl.is_active():
            self.stop_scan()
        if getattr(self, "_aiming_active", False):
            self.stop_aiming()

        self.ctrl.send({"cmd": "laser", "value": 0})
        self.laser_state = False

        if not self._scheduling_active:
            self._set_scheduling_ui_state(False)

    def _roundrobin_worker(self):
        mode_label = getattr(self, "_scheduling_mode_label", "RoundRobin")
        mode_key = getattr(self, "_scheduling_mode_key", "round_robin")
        is_proposed_mode = str(mode_key).strip().lower() == "proposed"
        final_message = f"✅ {mode_label} 완료"
        final_color = "green"
        scheduling_started_at = None
        rr_started_at = None
        T_total_sec = 0.0
        requested_total_sec = 0.0
        T_frame_sec = 0.0
        K_frames = None
        N_targets = 0
        t_slice_rr = 0.0
        completed_frames = 0
        loop_count = 0
        dwell_s = None
        csv_path = None
        active_scheduler = None
        proposed_state = None
        current_frame_allocations = {}
        current_frame_scores = {}
        try:
            scheduling_started_at = time.monotonic()
            final_targets = dict(getattr(self, "computed_targets", {}) or {})
            if hasattr(self, "_get_pointing_csv_path"):
                try:
                    csv_path = self._get_pointing_csv_path()
                except Exception:
                    csv_path = None
            settle_s = float(self._call_on_ui_thread(lambda: self.scan_tab.settle.get(), timeout=2.0))
            settle_s = max(0.1, settle_s)
            T_frame_sec_cfg = self._call_on_ui_thread(
                lambda: self.scheduling_tab.get_frame_seconds() if hasattr(self, "scheduling_tab") else ROUNDROBIN_T_FRAME_SEC_DEFAULT,
                timeout=2.0
            )
            T_total_sec_cfg = self._call_on_ui_thread(
                lambda: self.scheduling_tab.get_total_seconds() if hasattr(self, "scheduling_tab") else ROUNDROBIN_T_TOTAL_SEC_DEFAULT,
                timeout=2.0
            )
            led_probe_s = float(self._call_on_ui_thread(
                lambda: self.scheduling_tab.get_led_probe_seconds() if hasattr(self, "scheduling_tab") else 10.0,
                timeout=2.0
            ))
            led_probe_s = max(0.5, led_probe_s)
            battery_check_enabled = bool(self._call_on_ui_thread(
                lambda: self.scheduling_tab.get_battery_check_enabled() if hasattr(self, "scheduling_tab") else False,
                timeout=2.0,
            ))
            rr_battery_check_enabled = bool(battery_check_enabled)
            battery_sampling_enabled = True if is_proposed_mode else rr_battery_check_enabled
            self._scheduling_led_latest = {}
            self._scheduling_led_history = []
            self._update_scheduling_progress(
                scheduling_elapsed_s=0.0,
                phase="초기화",
                completed_frames=0,
                total_frames=None,
            )

            if final_targets:
                self._set_scheduling_status(
                    f"🔁 {mode_label}: 기존 타깃 {len(final_targets)}개 사용, 바로 시작",
                    fg="blue",
                )
                self._set_preview_overlay(current_id=None, phase="RR")
            else:
                self._set_scheduling_status(f"🔁 {mode_label}: Scan 시작", fg="blue")
                self._set_preview_overlay(current_id=None, phase="Scan")
                params = self._call_on_ui_thread(lambda: self.scan_tab.get_scan_params())
                self._call_on_ui_thread(lambda: setattr(self, "computed_targets", {}), timeout=2.0)
                self._call_on_ui_thread(lambda p=dict(params): self.start_scan(p), timeout=3.0)

                while not self._scheduling_stop_event.is_set():
                    self._update_scheduling_progress(
                        scheduling_elapsed_s=time.monotonic() - scheduling_started_at,
                        phase="Scan",
                    )
                    if self._scan_finished_event.wait(timeout=0.2):
                        break
                if self._scheduling_stop_event.is_set():
                    final_message = "⛔ Scheduling 중지됨"
                    final_color = "red"
                    return

                result = getattr(self, "_last_scan_result", None)
                csv_path = result.get("csv_path_abs") if result else None
                targets = dict(getattr(self, "computed_targets", {}) or {})

                # stop_scan에서 auto-compute 실패했을 때만 한 번 더 보정
                if not targets and csv_path:
                    self._set_scheduling_status(f"🔎 {mode_label}: CSV Compute 재시도...", fg="blue")
                    self._call_on_ui_thread(lambda p=csv_path: self.pointing_compute(p), timeout=60.0)
                    targets = dict(getattr(self, "computed_targets", {}) or {})

                if not targets:
                    raise RuntimeError("계산된 타깃이 없습니다. Scan/YOLO 결과를 확인하세요.")

                ordered_ids = self._order_scheduling_target_ids(targets)
                self._set_scheduling_status(
                    f"🎯 {mode_label}: {len(ordered_ids)}개 ID Adaptive 수렴 시작",
                    fg="blue",
                )

                # Scheduling에서는 IR 모드 강제 유지
                self._call_on_ui_thread(lambda: self.set_ir_cut("day"), timeout=3.0)
                time.sleep(0.1)

                # Phase A: 모든 ID를 adaptive aiming으로 수렴
                for idx, track_id in enumerate(ordered_ids, start=1):
                    if self._scheduling_stop_event.is_set():
                        final_message = "⛔ Scheduling 중지됨"
                        final_color = "red"
                        return

                    self._set_scheduling_status(
                        f"🎯 Adaptive [{idx}/{len(ordered_ids)}] ID {track_id} 수렴 중...",
                        fg="blue",
                    )
                    self._set_preview_overlay(
                        current_id=track_id,
                        phase="Adaptive",
                        led_state=self._scheduling_led_latest.get(track_id, "-"),
                    )
                    adaptive_started_at = time.monotonic()
                    self._update_scheduling_progress(
                        scheduling_elapsed_s=time.monotonic() - scheduling_started_at,
                        phase="Adaptive",
                        current_id=track_id,
                        current_elapsed_s=0.0,
                        current_total_s=ROUNDROBIN_AIM_TIMEOUT_S,
                    )

                    self._call_on_ui_thread(lambda: self.set_pointing_mode("adaptive"), timeout=2.0)
                    self._call_on_ui_thread(
                        lambda tid=track_id: self.move_to_target(
                            tid,
                            use_tilt_approach=False,
                            use_pan_tilt_approach=True,
                            pan_tilt_approach_wait_s=0.3,
                        ),
                        timeout=3.0,
                    )
                    time.sleep(settle_s)

                    started = bool(self._call_on_ui_thread(lambda tid=track_id: self.start_aiming(tid), timeout=3.0))
                    if not started:
                        print(f"[Scheduling] Adaptive start failed for ID {track_id}, skip")
                        continue

                    aim_deadline = time.monotonic() + ROUNDROBIN_AIM_TIMEOUT_S
                    while not self._scheduling_stop_event.is_set():
                        if not getattr(self, "_aiming_active", False):
                            break
                        adaptive_elapsed_s = time.monotonic() - adaptive_started_at
                        self._update_scheduling_progress(
                            scheduling_elapsed_s=time.monotonic() - scheduling_started_at,
                            phase="Adaptive",
                            current_id=track_id,
                            current_elapsed_s=min(adaptive_elapsed_s, ROUNDROBIN_AIM_TIMEOUT_S),
                            current_total_s=ROUNDROBIN_AIM_TIMEOUT_S,
                        )
                        if time.monotonic() >= aim_deadline:
                            print(f"[Scheduling] Adaptive timeout for ID {track_id} -> stop_aiming")
                            self.stop_aiming()
                            break
                        time.sleep(0.2)

                    if self._scheduling_stop_event.is_set():
                        final_message = "⛔ Scheduling 중지됨"
                        final_color = "red"
                        return

                    # 다음 ID 전환 전 레이저 강제 OFF
                    self.ctrl.send({"cmd": "laser", "value": 0})
                    self.laser_state = False
                    time.sleep(0.2)
                    # adaptive 중 수집된 최신 LED 예측이 있으면 반영
                    led_hint = getattr(self, "_last_object_led_info", {}).get("pred")
                    if led_hint:
                        self._scheduling_led_latest[track_id] = str(led_hint)
                    roi_hint = getattr(self, "_last_object_led_info", {}).get("roi")
                    if roi_hint is not None:
                        try:
                            self._track_led_roi[track_id] = tuple(int(v) for v in roi_hint)
                        except Exception:
                            pass

                final_targets = dict(getattr(self, "computed_targets", {}) or {})
                if not final_targets:
                    raise RuntimeError("Adaptive 수렴 후 사용 가능한 ID가 없습니다.")

            final_ids = self._order_scheduling_target_ids(final_targets)
            if not final_ids:
                raise RuntimeError("Scheduling용 타깃이 없습니다.")

            rr_timing = self._resolve_roundrobin_timing(
                n_targets=len(final_ids),
                frame_sec=T_frame_sec_cfg,
                total_sec=T_total_sec_cfg,
            )
            self._roundrobin_timing = dict(rr_timing)
            T_frame_sec = rr_timing["T_frame_sec"]
            T_total_sec = rr_timing["T_total_sec"]
            requested_total_sec = rr_timing.get("T_total_sec_requested", T_total_sec)
            K_frames = rr_timing["K_frames"]
            N_targets = rr_timing["N_targets"]
            t_slice_rr = rr_timing["t_slice_rr"]
            dwell_s = rr_timing["dwell_s"]
            active_scheduler = getattr(self._get_scheduling_backend(), "scheduler", None)

            k_text = str(K_frames) if K_frames is not None else "manual"
            if T_total_sec > 0.0:
                total_text = f"{T_total_sec:.1f}s"
                if requested_total_sec > 0.0 and abs(T_total_sec - requested_total_sec) > 1e-6:
                    total_text = f"{T_total_sec:.1f}s (req {requested_total_sec:.1f}s)"
            else:
                total_text = "manual"

            if is_proposed_mode and isinstance(active_scheduler, ProposedScheduler):
                initial_led_states = {}
                final_led_state_map = dict(getattr(self, "_track_final_led_state", {}) or {})
                for track_id in final_ids:
                    state_info = dict(final_led_state_map.get(track_id) or {})
                    initial_led_states[track_id] = (
                        state_info.get("bits")
                        or state_info.get("pred")
                        or self._scheduling_led_latest.get(track_id)
                        or "000"
                    )
                proposed_state = active_scheduler.initialize_state(
                    total_frame_time=T_frame_sec,
                    ordered_track_ids=final_ids,
                    csv_path=csv_path,
                    track_id_members=dict(getattr(self, "_pointing_csv_track_ids", {}) or {}),
                    initial_led_states=initial_led_states,
                )
                frame_plan = active_scheduler.build_frame_plan(proposed_state)
                current_frame_allocations = dict(frame_plan.get("allocations") or {})
                current_frame_scores = dict(frame_plan.get("scores") or {})
                self._set_scheduling_frame_debug(
                    mode_label=mode_label,
                    execution_order=list(final_ids),
                    frame_allocations=dict(current_frame_allocations),
                    fixed_target_coeffs={},
                    battery_state_prev=dict(frame_plan.get("battery_state_prev") or {}),
                    battery_coeff_prev=dict(frame_plan.get("battery_coeff_prev") or {}),
                    frame_scores=dict(current_frame_scores),
                )
                battery_check_text = "Battery update end-sample"
            else:
                current_frame_allocations = {int(track_id): float(t_slice_rr) for track_id in final_ids}
                current_frame_scores = {int(track_id): 1.0 for track_id in final_ids}
                initial_state_debug = {}
                initial_coeff_debug = {}
                final_led_state_map = dict(getattr(self, "_track_final_led_state", {}) or {})
                for track_id in final_ids:
                    state_info = dict(final_led_state_map.get(track_id) or {})
                    led_state = state_info.get("pred") or self._scheduling_led_latest.get(track_id) or "R"
                    initial_state_debug[int(track_id)] = str(led_state)
                    initial_coeff_debug[int(track_id)] = float(led_state_to_battery_coeff(led_state))
                self._set_scheduling_frame_debug(
                    mode_label=mode_label,
                    execution_order=list(final_ids),
                    frame_allocations=dict(current_frame_allocations),
                    fixed_target_coeffs={int(track_id): 1.0 for track_id in final_ids},
                    battery_state_prev=initial_state_debug,
                    battery_coeff_prev=initial_coeff_debug,
                    frame_scores=dict(current_frame_scores),
                )
                battery_check_text = (
                    f"Battery check {led_probe_s:.1f}초"
                    if rr_battery_check_enabled
                    else "Battery check off"
                )
            alloc_text = (
                "dynamic allocation"
                if is_proposed_mode
                else f"slice={t_slice_rr:.1f}초/ID"
            )
            self._set_scheduling_status(
                f"🔴 {mode_label}: {len(final_ids)}개 ID 순환 조사 시작 "
                f"({alloc_text}, T_frame={T_frame_sec:.1f}s, "
                f"T_total={total_text}, K={k_text}, N={N_targets}, "
                f"{battery_check_text})",
                fg="blue",
            )
            self._set_preview_overlay(current_id=None, phase="RR", dwell_elapsed=0.0, dwell_total=dwell_s)

            # Shoot loop에서 preview 강제 ON
            preview_on = bool(self._call_on_ui_thread(lambda: self.preview_active, timeout=2.0))
            if not preview_on:
                self._call_on_ui_thread(
                    lambda: self.toggle_preview(True, *self._get_preview_cfg()),
                    timeout=4.0,
                )
                time.sleep(0.2)

            # RR / Proposed phase keeps laser continuously ON so frame timing stays absolute.
            self._call_on_ui_thread(lambda: self.set_ir_cut("day"), timeout=3.0)
            self.ctrl.send({"cmd": "laser", "value": 1})
            self.laser_state = True

            rr_started_at = time.monotonic()
            completed_frames = 0
            self._update_scheduling_progress(
                scheduling_elapsed_s=time.monotonic() - scheduling_started_at,
                shoot_elapsed_s=0.0,
                shoot_total_s=T_total_sec if T_total_sec > 0.0 else None,
                phase="RR 준비",
                loop_count=1,
                frame_elapsed_s=0.0,
                frame_total_s=T_frame_sec,
                slice_total=N_targets,
                current_elapsed_s=0.0,
                current_total_s=dwell_s,
                completed_frames=completed_frames,
                total_frames=K_frames,
            )

            # Phase B: ID 순환 조사 (Stop까지 반복)
            loop_count = 0
            while not self._scheduling_stop_event.is_set():
                if K_frames is not None and completed_frames >= K_frames:
                    final_message = f"✅ {mode_label} 완료 (K_frames 도달)"
                    final_color = "green"
                    break
                loop_count = completed_frames + 1
                if is_proposed_mode and proposed_state is not None and isinstance(active_scheduler, ProposedScheduler):
                    frame_plan = active_scheduler.build_frame_plan(
                        proposed_state,
                        total_frame_time=T_frame_sec,
                        execution_order=final_ids,
                    )
                    current_frame_allocations = dict(frame_plan.get("allocations") or {})
                    current_frame_scores = dict(frame_plan.get("scores") or {})
                    self._set_scheduling_frame_debug(
                        mode_label=mode_label,
                        execution_order=list(final_ids),
                        frame_allocations=dict(current_frame_allocations),
                        fixed_target_coeffs={},
                        battery_state_prev=dict(frame_plan.get("battery_state_prev") or {}),
                        battery_coeff_prev=dict(frame_plan.get("battery_coeff_prev") or {}),
                        frame_scores=dict(current_frame_scores),
                    )
                else:
                    current_frame_allocations = {int(track_id): float(t_slice_rr) for track_id in final_ids}
                    current_frame_scores = {int(track_id): 1.0 for track_id in final_ids}
                    self._set_scheduling_frame_debug(
                        mode_label=mode_label,
                        execution_order=list(final_ids),
                        frame_allocations=dict(current_frame_allocations),
                        frame_scores=dict(current_frame_scores),
                    )
                frame_started_at = rr_started_at + (completed_frames * T_frame_sec)
                frame_deadline = frame_started_at + T_frame_sec

                self._update_scheduling_progress(
                    scheduling_elapsed_s=time.monotonic() - scheduling_started_at,
                    shoot_elapsed_s=min(time.monotonic() - rr_started_at, completed_frames * T_frame_sec),
                    shoot_total_s=T_total_sec if T_total_sec > 0.0 else None,
                    phase="Frame 할당 준비",
                    loop_count=loop_count,
                    frame_elapsed_s=0.0,
                    frame_total_s=T_frame_sec,
                    slice_total=N_targets,
                    current_elapsed_s=0.0,
                    current_total_s=None,
                    completed_frames=completed_frames,
                    total_frames=K_frames,
                )

                if not self._wait_until_scheduling_deadline(frame_started_at):
                    break

                slice_offset_s = 0.0
                for idx, track_id in enumerate(final_ids, start=1):
                    if self._scheduling_stop_event.is_set():
                        break
                    alloc_s = max(0.0, float(current_frame_allocations.get(track_id, t_slice_rr)))
                    slice_start = frame_started_at + slice_offset_s
                    slice_end = frame_started_at + slice_offset_s + alloc_s
                    slice_offset_s += alloc_s

                    if not self._wait_until_scheduling_deadline(slice_start):
                        break
                    if time.monotonic() >= slice_end:
                        continue

                    # Shoot loop 중 preview가 꺼졌다면 즉시 복구
                    preview_on_loop = bool(self._call_on_ui_thread(lambda: self.preview_active, timeout=2.0))
                    if not preview_on_loop:
                        self._call_on_ui_thread(
                            lambda: self.toggle_preview(True, *self._get_preview_cfg()),
                            timeout=4.0,
                        )
                        if not self._wait_until_scheduling_deadline(min(slice_end, time.monotonic() + 0.1)):
                            break
                    if time.monotonic() >= slice_end:
                        continue

                    self._set_scheduling_status(
                        f"🔴 {mode_label} frame {loop_count} [{idx}/{len(final_ids)}] ID {track_id}",
                        fg="blue",
                    )
                    led_state = self._scheduling_led_latest.get(track_id, "-")
                    self._update_scheduling_progress(
                        scheduling_elapsed_s=time.monotonic() - scheduling_started_at,
                        shoot_elapsed_s=min(time.monotonic() - rr_started_at, (completed_frames * T_frame_sec) + T_frame_sec),
                        shoot_total_s=T_total_sec if T_total_sec > 0.0 else None,
                        phase="RR",
                        current_id=track_id,
                        loop_count=loop_count,
                        frame_elapsed_s=min(max(0.0, time.monotonic() - frame_started_at), T_frame_sec),
                        frame_total_s=T_frame_sec,
                        slice_index=idx,
                        slice_total=N_targets,
                        current_elapsed_s=min(max(0.0, time.monotonic() - slice_start), alloc_s),
                        current_total_s=alloc_s,
                        completed_frames=completed_frames,
                        total_frames=K_frames,
                    )
                    self._set_preview_overlay(
                        current_id=track_id,
                        phase="RR",
                        dwell_elapsed=min(max(0.0, time.monotonic() - slice_start), alloc_s),
                        dwell_total=alloc_s,
                        led_state=led_state,
                    )
                    self._call_on_ui_thread(
                        lambda tid=track_id, use_special=(idx == 1), wait_s=ROUNDROBIN_FIRST_APPROACH_WAIT_S: self.move_to_target(
                            tid,
                            use_tilt_approach=False,
                            use_pan_tilt_approach=use_special,
                            pan_tilt_approach_wait_s=wait_s,
                        ),
                        timeout=5.0,
                    )
                    if time.monotonic() >= slice_end:
                        continue

                    # Per-slice IR policy: if slice < 3s, use first half only; otherwise 3s.
                    slice_ir_head_s = 3.0 if alloc_s >= 3.0 else max(0.0, alloc_s * 0.5)
                    current_mode = "day"  # day=IR mode, night=Normal mode
                    self._call_on_ui_thread(lambda m=current_mode: self.set_ir_cut(m), timeout=3.0)

                    # Battery update/check는 slice budget 내부에서만 허용
                    next_probe_elapsed = led_probe_s if rr_battery_check_enabled else None
                    proposed_sample_index = 0
                    proposed_sample_elapsed_points = []
                    if is_proposed_mode and proposed_state is not None and isinstance(active_scheduler, ProposedScheduler):
                        proposed_sample_elapsed_points = list(active_scheduler.get_sampling_elapsed_points(alloc_s, interval_s=10.0))
                    edge_eps = 1e-3  # 경계(시작/끝) 체크 제외용
                    while time.monotonic() < slice_end:
                        if self._scheduling_stop_event.is_set():
                            break
                        now = time.monotonic()
                        slice_elapsed = min(max(0.0, now - slice_start), alloc_s)
                        while (
                            is_proposed_mode
                            and proposed_state is not None
                            and isinstance(active_scheduler, ProposedScheduler)
                            and proposed_sample_index < len(proposed_sample_elapsed_points)
                            and slice_elapsed >= proposed_sample_elapsed_points[proposed_sample_index]
                        ):
                            remaining_for_probe = slice_end - time.monotonic()
                            sampled_led_state = None
                            probe_timeout = min(remaining_for_probe - 0.05, 10.0)
                            if probe_timeout > 0.2:
                                sampled_led_state = self._probe_led_state_for_track(
                                    track_id,
                                    probe_interval_s=10.0,
                                    timeout_s=probe_timeout,
                                    return_bits=True,
                                    refresh_roi=True,
                                )
                            update = active_scheduler.sample_or_update_battery_state_for_target(
                                proposed_state,
                                track_id,
                                sampled_led_state,
                            )
                            self._scheduling_led_latest[track_id] = str(update.get("next_state", "NONE"))
                            proposed_sample_index += 1
                            if probe_timeout <= 0.2:
                                break
                        while (
                            rr_battery_check_enabled
                            and next_probe_elapsed is not None
                            and slice_elapsed >= next_probe_elapsed
                        ):
                            probe_elapsed = next_probe_elapsed
                            next_probe_elapsed += led_probe_s

                            # "시작/끝 제외": 현재 ID 구간 내부 시점에서만 체크
                            if probe_elapsed <= edge_eps or probe_elapsed >= (alloc_s - edge_eps):
                                continue
                            remaining_for_probe = slice_end - time.monotonic()
                            probe_timeout = min(remaining_for_probe - 0.1, led_probe_s)
                            if probe_timeout <= 0.2:
                                break

                            probe_pred = self._probe_led_state_for_track(
                                track_id,
                                probe_interval_s=led_probe_s,
                                timeout_s=probe_timeout,
                            )
                            self._scheduling_led_latest[track_id] = probe_pred
                            # 한 루프에서 과도한 연속 체크 방지
                            break

                        desired_mode = "day" if slice_elapsed < slice_ir_head_s else "night"
                        if desired_mode != current_mode:
                            self._call_on_ui_thread(lambda m=desired_mode: self.set_ir_cut(m), timeout=3.0)
                            current_mode = desired_mode
                        self._update_scheduling_progress(
                            scheduling_elapsed_s=time.monotonic() - scheduling_started_at,
                            shoot_elapsed_s=min(time.monotonic() - rr_started_at, (completed_frames * T_frame_sec) + T_frame_sec),
                            shoot_total_s=T_total_sec if T_total_sec > 0.0 else None,
                            phase="RR",
                            current_id=track_id,
                            loop_count=loop_count,
                            frame_elapsed_s=min(max(0.0, time.monotonic() - frame_started_at), T_frame_sec),
                            frame_total_s=T_frame_sec,
                            slice_index=idx,
                            slice_total=N_targets,
                            current_elapsed_s=slice_elapsed,
                            current_total_s=alloc_s,
                            completed_frames=completed_frames,
                            total_frames=K_frames,
                        )
                        self._set_preview_overlay(
                            current_id=track_id,
                            phase="RR",
                            dwell_elapsed=slice_elapsed,
                            dwell_total=alloc_s,
                            led_state=self._scheduling_led_latest.get(track_id, "-"),
                        )
                        if not self._wait_until_scheduling_deadline(min(slice_end, time.monotonic() + 0.1)):
                            break

                    if (
                        is_proposed_mode
                        and proposed_state is not None
                        and isinstance(active_scheduler, ProposedScheduler)
                        and track_id not in dict(proposed_state.get("battery_state_next", {}) or {})
                    ):
                        update = active_scheduler.sample_or_update_battery_state_for_target(
                            proposed_state,
                            track_id,
                            None,
                        )
                        self._scheduling_led_latest[track_id] = str(update.get("next_state", "NONE"))

                    if self._scheduling_stop_event.is_set():
                        break
                    if not self._wait_until_scheduling_deadline(slice_end):
                        break

                if self._scheduling_stop_event.is_set():
                    break
                if not self._wait_until_scheduling_deadline(frame_deadline):
                    break

                if is_proposed_mode and proposed_state is not None and isinstance(active_scheduler, ProposedScheduler):
                    active_scheduler.log_frame_summary(proposed_state)
                    active_scheduler.finalize_frame_and_prepare_next(proposed_state)

                completed_frames += 1
                self._update_scheduling_progress(
                    scheduling_elapsed_s=time.monotonic() - scheduling_started_at,
                    shoot_elapsed_s=completed_frames * T_frame_sec,
                    shoot_total_s=T_total_sec if T_total_sec > 0.0 else None,
                    phase="프레임 완료",
                    loop_count=loop_count,
                    frame_elapsed_s=T_frame_sec,
                    frame_total_s=T_frame_sec,
                    completed_frames=completed_frames,
                    total_frames=K_frames,
                )

            if self._scheduling_stop_event.is_set():
                final_message = "⛔ Scheduling 중지됨"
                final_color = "red"

        except Exception as e:
            final_message = f"❌ {mode_label} 오류: {e}"
            final_color = "red"
            print(final_message)
        finally:
            try:
                self.ctrl.send({"cmd": "laser", "value": 0})
                self.laser_state = False
            except Exception:
                pass
            if scheduling_started_at is not None:
                phase_label = "완료" if final_color == "green" else "중지" if "중지" in final_message else "오류"
                end_now = time.monotonic()
                self._update_scheduling_progress(
                    scheduling_elapsed_s=end_now - scheduling_started_at,
                    shoot_elapsed_s=(
                        min(
                            max(0.0, end_now - rr_started_at),
                            T_total_sec if T_total_sec > 0.0 else max(0.0, end_now - rr_started_at),
                        )
                        if rr_started_at is not None and T_frame_sec > 0.0
                        else None
                    ),
                    shoot_total_s=T_total_sec if T_total_sec > 0.0 else None,
                    phase=phase_label,
                    loop_count=loop_count if loop_count > 0 else None,
                    frame_elapsed_s=(
                        min(
                            max(
                                0.0,
                                end_now - (rr_started_at + (max(0, loop_count - 1) * T_frame_sec)),
                            ),
                            T_frame_sec,
                        )
                        if rr_started_at is not None and loop_count > 0 and T_frame_sec > 0.0
                        else None
                    ),
                    frame_total_s=T_frame_sec if T_frame_sec > 0.0 else None,
                    completed_frames=completed_frames,
                    total_frames=K_frames,
                )
            self.root.after(0, lambda msg=final_message, fg=final_color: self._finalize_scheduling_ui(msg, fg))
    
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

    def set_preview_crosshair(self, visible):
        """Live Preview 중앙 십자가 표시 토글"""
        if hasattr(self, "preview_frame") and hasattr(self.preview_frame, "set_crosshair_visible"):
            self.preview_frame.set_crosshair_visible(bool(visible))
    
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
