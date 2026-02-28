#!/usr/bin/env python3
"""
Com Client - Modular architecture with mixins
"""

import pathlib
import datetime
import time
from tkinter import Tk, Label, Frame, ttk
import cv2
import numpy as np

from network import GuiCtrlClient, GuiImgClient
from event_handlers import EventHandlersMixin
from pointing_handler import PointingHandlerMixin
from app_helpers import AppHelpersMixin
from ui_components import PreviewFrame, ScanTab, TestSettingsTab, PointingTab, SchedulingTab
from scan_controller import ScanController
from yolo_utils import YOLOProcessor
from led_filter import classify_from_single_roi, get_default_led_filter_params
import threading

SERVER_HOST = "127.0.0.1"
GUI_CTRL_PORT = 7600
GUI_IMG_PORT = 7601

# 저장 디렉토리
SAVE_DIR = pathlib.Path("captures")
ROUNDROBIN_DWELL_S = 20.0
ROUNDROBIN_AIM_TIMEOUT_S = 120.0


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
        tab_scheduling = Frame(self.notebook)
        
        self.notebook.add(tab_scan, text="Scan")
        self.notebook.add(tab_test, text="Test & Settings")
        self.notebook.add(tab_pointing, text="Pointing")
        self.notebook.add(tab_scheduling, text="Scheduling")
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
            "stop_scheduling": self.stop_scheduling,
        }
        self.scheduling_tab = SchedulingTab(tab_scheduling, scheduling_callbacks)
        
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

        # Pointing related initializations (from PointingHandlerMixin)
        self._pointing_gains = {}
        self._pointing_img_event = threading.Event()
        self.pointing_mode = "adaptive"
        
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

    def _set_preview_overlay(self, current_id=None, phase="Idle", dwell_elapsed=None, dwell_total=None, led_state=None):
        """프리뷰 오버레이(현재 ID + Phase + Shoot Dwell 진행률) 갱신"""
        cid = "-" if current_id is None else str(current_id)
        led_txt = "-" if led_state in (None, "") else str(led_state)
        if dwell_elapsed is not None and dwell_total is not None and dwell_total > 0:
            text = f"Shoot Timer: {dwell_elapsed:.1f}/{dwell_total:.1f}s | ID: {cid} | LED: {led_txt} | {phase}"
        else:
            text = f"Shoot Timer: - | ID: {cid} | LED: {led_txt} | {phase}"
        if hasattr(self, "preview_frame") and hasattr(self.preview_frame, "set_overlay_text"):
            self.root.after(0, lambda t=text: self.preview_frame.set_overlay_text(t))

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
                if self._scheduling_stop_event.is_set():
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

    def _probe_led_state_for_track(self, track_id, probe_interval_s=2.0):
        """
        Scheduling shoot loop 중 K초 주기 LED 상태 프로브.
        LED ON/OFF 없이, 저장된 ROI에서 단일 프레임으로 LED 상태를 판정한다.
        카메라 모드는 shoot loop의 현재 상태를 그대로 유지한다.
        """
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        snap_name = f"sched_led_single_id{track_id}_{ts}.jpg"

        preview_was_on = bool(self._call_on_ui_thread(lambda: self.preview_active, timeout=2.0))
        pred = "NONE"
        score = {"R": 0, "G": 0, "B": 0}
        roi = None
        try:
            roi = self._track_led_roi.get(track_id)
            if roi is None:
                print(f"[Scheduling] LED probe skipped (ID {track_id}): no stored ROI")
                return "NONE"

            img = self._blocking_snap_and_wait(snap_name, timeout=10.0, shutter_speed=10000, analogue_gain=None)
            if img is None:
                return "NONE"

            pred, score, roi_used = classify_from_single_roi(
                img,
                roi,
                params=self.led_filter_params,
            )
            if roi_used is not None:
                self._track_led_roi[track_id] = tuple(int(v) for v in roi_used)
            roi = roi_used

            self._scheduling_led_latest[track_id] = pred
            self._scheduling_led_history.append({
                "ts": ts,
                "track_id": int(track_id),
                "pred": pred,
                "r": int(score["R"]),
                "g": int(score["G"]),
                "b": int(score["B"]),
                "roi": tuple(int(v) for v in roi) if roi is not None else None,
                "mode": "single_roi",
                "probe_interval_s": float(probe_interval_s),
            })
            return pred
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
        if hasattr(self, "scheduling_tab"):
            self.scheduling_tab.set_running_state(False)
            self.scheduling_tab.update_status(message, fg=fg)
        self.info_label.config(text=message)
        self._set_preview_overlay(current_id=None, phase="Idle")

    def start_roundrobin(self):
        """RoundRobin 스케줄 시작"""
        if self._scheduling_active:
            self._set_scheduling_status("⚠️ Scheduling already running", fg="orange")
            return False
        if self.scan_ctrl.is_active():
            self._set_scheduling_status("⚠️ Scan already running", fg="orange")
            return False
        if getattr(self, "_aiming_active", False):
            self._set_scheduling_status("⚠️ Pointing is running. Stop aiming first.", fg="orange")
            return False

        self._scheduling_active = True
        self._scheduling_stop_event.clear()
        self._set_scheduling_ui_state(True)
        self._set_scheduling_status("🔁 RoundRobin 시작...", fg="blue")

        self._scheduling_thread = threading.Thread(target=self._roundrobin_worker, daemon=True)
        self._scheduling_thread.start()
        return True

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
        final_message = "✅ RoundRobin 완료"
        final_color = "green"
        try:
            final_targets = dict(getattr(self, "computed_targets", {}) or {})
            settle_s = float(self._call_on_ui_thread(lambda: self.scan_tab.settle.get(), timeout=2.0))
            settle_s = max(0.1, settle_s)
            dwell_s = float(self._call_on_ui_thread(
                lambda: self.scheduling_tab.get_dwell_seconds() if hasattr(self, "scheduling_tab") else ROUNDROBIN_DWELL_S,
                timeout=2.0
            ))
            dwell_s = max(0.2, dwell_s)
            led_probe_s = float(self._call_on_ui_thread(
                lambda: self.scheduling_tab.get_led_probe_seconds() if hasattr(self, "scheduling_tab") else 10.0,
                timeout=2.0
            ))
            led_probe_s = max(0.5, led_probe_s)
            self._scheduling_led_latest = {}
            self._scheduling_led_history = []

            if final_targets:
                self._set_scheduling_status(
                    f"🔁 RoundRobin: 기존 타깃 {len(final_targets)}개 사용, Shoot 바로 시작",
                    fg="blue",
                )
                self._set_preview_overlay(current_id=None, phase="Shoot")
            else:
                self._set_scheduling_status("🔁 RoundRobin: Scan 시작", fg="blue")
                self._set_preview_overlay(current_id=None, phase="Scan")
                params = self._call_on_ui_thread(lambda: self.scan_tab.get_scan_params())
                self._call_on_ui_thread(lambda: setattr(self, "computed_targets", {}), timeout=2.0)
                self._call_on_ui_thread(lambda p=dict(params): self.start_scan(p), timeout=3.0)

                while not self._scheduling_stop_event.is_set():
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
                    self._set_scheduling_status("🔎 RoundRobin: CSV Compute 재시도...", fg="blue")
                    self._call_on_ui_thread(lambda p=csv_path: self.pointing_compute(p), timeout=60.0)
                    targets = dict(getattr(self, "computed_targets", {}) or {})

                if not targets:
                    raise RuntimeError("계산된 타깃이 없습니다. Scan/YOLO 결과를 확인하세요.")

                ordered_ids = sorted(targets.keys())
                self._set_scheduling_status(
                    f"🎯 RoundRobin: {len(ordered_ids)}개 ID Adaptive 수렴 시작",
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

                    self._call_on_ui_thread(lambda: self.set_pointing_mode("adaptive"), timeout=2.0)
                    self._call_on_ui_thread(lambda tid=track_id: self.move_to_target(tid), timeout=3.0)
                    time.sleep(settle_s)

                    started = bool(self._call_on_ui_thread(lambda tid=track_id: self.start_aiming(tid), timeout=3.0))
                    if not started:
                        print(f"[Scheduling] Adaptive start failed for ID {track_id}, skip")
                        continue

                    aim_deadline = time.monotonic() + ROUNDROBIN_AIM_TIMEOUT_S
                    while not self._scheduling_stop_event.is_set():
                        if not getattr(self, "_aiming_active", False):
                            break
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

            final_ids = sorted(final_targets.keys())
            if not final_ids:
                raise RuntimeError("Scheduling용 타깃이 없습니다.")

            self._set_scheduling_status(
                f"🔴 RoundRobin Shoot: {len(final_ids)}개 ID 순환 조사 시작 "
                f"({dwell_s:.1f}초/ID, Battery check {led_probe_s:.1f}초, Stop까지 계속)",
                fg="blue",
            )
            self._set_preview_overlay(current_id=None, phase="Shoot", dwell_elapsed=0.0, dwell_total=dwell_s)

            # Shoot loop에서 preview 강제 ON
            preview_on = bool(self._call_on_ui_thread(lambda: self.preview_active, timeout=2.0))
            if not preview_on:
                self._call_on_ui_thread(
                    lambda: self.toggle_preview(True, *self._get_preview_cfg()),
                    timeout=4.0,
                )
                time.sleep(0.2)
            
            # Phase B: ID 순환 조사 (Stop까지 반복)
            loop_count = 0
            while not self._scheduling_stop_event.is_set():
                loop_count += 1
                # 루프마다 IR 모드 재보장
                self._call_on_ui_thread(lambda: self.set_ir_cut("day"), timeout=3.0)
                for idx, track_id in enumerate(final_ids, start=1):
                    if self._scheduling_stop_event.is_set():
                        break

                    # Shoot loop 중 preview가 꺼졌다면 즉시 복구
                    preview_on_loop = bool(self._call_on_ui_thread(lambda: self.preview_active, timeout=2.0))
                    if not preview_on_loop:
                        self._call_on_ui_thread(
                            lambda: self.toggle_preview(True, *self._get_preview_cfg()),
                            timeout=4.0,
                        )
                        time.sleep(0.1)

                    self._set_scheduling_status(
                        f"🔴 Shoot loop {loop_count} [{idx}/{len(final_ids)}] ID {track_id}",
                        fg="blue",
                    )
                    led_state = self._scheduling_led_latest.get(track_id, "-")
                    self._set_preview_overlay(
                        current_id=track_id,
                        phase="Shoot",
                        dwell_elapsed=0.0,
                        dwell_total=dwell_s,
                        led_state=led_state,
                    )
                    self._call_on_ui_thread(
                        lambda tid=track_id: self.move_to_target(tid, use_tilt_approach=True),
                        timeout=5.0,
                    )
                    time.sleep(settle_s)

                    # Per-ID policy: first few seconds in IR, then keep Normal mode.
                    ir_head_s = 3.0
                    current_mode = "day"  # day=IR mode, night=Normal mode
                    self._call_on_ui_thread(lambda m=current_mode: self.set_ir_cut(m), timeout=3.0)

                    self.ctrl.send({"cmd": "laser", "value": 1})
                    self.laser_state = True

                    start_t = time.monotonic()
                    # Battery check 기준은 각 ID shoot elapsed(=start_t 기준)로 통일
                    next_probe_elapsed = led_probe_s
                    edge_eps = 1e-3  # 경계(시작/끝) 체크 제외용
                    while (time.monotonic() - start_t) < dwell_s:
                        if self._scheduling_stop_event.is_set():
                            break
                        elapsed = time.monotonic() - start_t
                        while elapsed >= next_probe_elapsed:
                            probe_elapsed = next_probe_elapsed
                            next_probe_elapsed += led_probe_s

                            # "시작/끝 제외": 현재 ID 구간 내부 시점에서만 체크
                            if probe_elapsed <= edge_eps or probe_elapsed >= (dwell_s - edge_eps):
                                continue

                            probe_pred = self._probe_led_state_for_track(
                                track_id,
                                probe_interval_s=led_probe_s,
                            )
                            self._scheduling_led_latest[track_id] = probe_pred
                            # 한 루프에서 과도한 연속 체크 방지
                            break

                        desired_mode = "day" if elapsed < ir_head_s else "night"
                        if desired_mode != current_mode:
                            self._call_on_ui_thread(lambda m=desired_mode: self.set_ir_cut(m), timeout=3.0)
                            current_mode = desired_mode
                        self._set_preview_overlay(
                            current_id=track_id,
                            phase="Shoot",
                            dwell_elapsed=min(elapsed, dwell_s),
                            dwell_total=dwell_s,
                            led_state=self._scheduling_led_latest.get(track_id, "-"),
                        )
                        time.sleep(0.1)

                    self.ctrl.send({"cmd": "laser", "value": 0})
                    self.laser_state = False
                    time.sleep(0.05)

            final_message = "⛔ Scheduling 중지됨"
            final_color = "red"

        except Exception as e:
            final_message = f"❌ RoundRobin 오류: {e}"
            final_color = "red"
            print(final_message)
        finally:
            try:
                self.ctrl.send({"cmd": "laser", "value": 0})
                self.laser_state = False
            except Exception:
                pass
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
