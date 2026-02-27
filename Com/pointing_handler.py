"""
Pointing mode handler mixin
Handles CSV analysis, target computation, and laser fine-aiming
"""

import csv
import time
import threading
import numpy as np
import cv2
from tkinter import filedialog
from collections import defaultdict
import datetime
import os


# ========== Constants ==========
CENTERING_GAIN_PAN = 0.03    # deg/px (기본값, 역산으로 대체됨)
CENTERING_GAIN_TILT = 0.03   # deg/px
CONVERGENCE_TOL_PX_X = 7       # 수렴 판정 임계값 X (px)
CONVERGENCE_TOL_PX_Y = 25      # 수렴 판정 임계값 Y (px)
OBJECT_SIZE_CM = 5.5         # 객체 크기 (cm) - offset 계산용
TARGET_OFFSET_CM = -12.25    # 객체 중심 아래 12.25cm (2.75 + 5.5 + 4)
LASER_DIFF_THRESHOLD = 150   # 레이저 diff threshold (산란광 제거)
MAX_STEP_DEG = 5.0           # 최대 보정 각도 (deg/step)
ROUGH_CAM_TO_LASER_CM = 6.0  # 카메라 기준 레이저 수직 오프셋 (cm)
ROUGH_TARGET_BELOW_CM = 12.5 # YOLO 인식 중심 아래 타겟 오프셋 (cm)
ROUGH_PHASE1_TOL_X_PX = 15  # Rough Phase 1 X 수렴 임계값 (px)
ROUGH_PHASE1_TOL_Y_PX = 15  # Rough Phase 1 Y 수렴 임계값 (px)
ROUGH_PHASE1_STUCK_WINDOW = 8        # Phase 1 진동 감지 윈도우(2-level repeat)
ROUGH_PHASE2_START_TILT_UP_DEG = 2.0  # Phase 2 시작 전 기본 상향 오프셋
ROUGH_PHASE2_TILT_STEP_DEG = 1.0      # Phase 2 tilt 탐색 스텝 (하강)
ROUGH_PHASE2_DROP_RATIO = 0.65        # 현재/이전 밝기 비율이 이 값 이하이면 급락 판정
ROUGH_PHASE2_DROP_DELTA = 8.0         # 이전-현재 평균 밝기 절대 감소량 임계
FINAL_TILT_APPROACH_UP_DEG = 1.0      # 최종 조준 시 tilt를 +1 후 원래 값으로 내리는 트릭


class PointingHandlerMixin:
    """Pointing mode logic - CSV analysis, regression, and laser fine-aiming"""
    
    # ========== CSV & Regression ==========

    def pointing_choose_csv(self):
        """CSV 파일 선택"""
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.point_csv_path.set(path)
            print(f"[Pointing] CSV selected: {path}")
    
    def pointing_compute(self, csv_path=None):
        """
        CSV를 읽어 Track ID별 회귀분석:
          1) Tilt별 cx = a*pan + b → pan_center = (W/2 - b)/a
          2) Pan별 cy = e*tilt + f → tilt_center = (H/2 - f)/e
        가중평균하여 최종 타깃 pan/tilt 계산
        
        Args:
            csv_path: (Optional) 직접 지정할 CSV 경로. 없으면 GUI 변수값 사용.
        """
        if csv_path:
            path = csv_path
            self.point_csv_path.set(path)  # GUI 변수도 업데이트
        else:
            path = self.point_csv_path.get().strip()
            
        if not path:
            print("[Pointing] CSV를 선택하세요")
            return
        
        try:
            rows = []
            W_frame = H_frame = None
            conf_min = 0.5  # Minimum confidence
            min_samples = 2  # Minimum samples for regression
            
            # CSV 읽기
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for d in reader:
                    if d.get("conf", "") == "":
                        continue
                    conf = float(d["conf"])
                    if conf < conf_min:
                        continue
                    
                    pan = d.get("pan_deg")
                    tilt = d.get("tilt_deg")
                    if pan in ("", None) or tilt in ("", None):
                        continue
                    
                    pan = float(pan)
                    tilt = float(tilt)
                    cx = float(d["cx"])
                    cy = float(d["cy"])
                    W = int(d["W"]) if d.get("W") else None
                    H = int(d["H"]) if d.get("H") else None
                    
                    # Track ID 파싱
                    track_id = int(d.get("track_id", 0))
                    
                    if W_frame is None and W:
                        W_frame = W
                    if H_frame is None and H:
                        H_frame = H
                    
                    rows.append({
                        'track_id': track_id,
                        'pan': pan,
                        'tilt': tilt,
                        'cx': cx,
                        'cy': cy
                    })
            
            if not rows:
                print("[Pointing] CSV에서 조건을 만족하는 행이 없습니다")
                return
            if W_frame is None or H_frame is None:
                print("[Pointing] CSV에 W/H 정보가 없습니다")
                return
            
            # Track ID별로 그룹화
            grouped_by_track = defaultdict(list)
            for row in rows:
                grouped_by_track[row['track_id']].append(row)
            
            print(f"[Pointing] Found {len(grouped_by_track)} track(s): {list(grouped_by_track.keys())}")
            
            # 각 track_id별로 독립적으로 계산
            self.computed_targets = {}  # {track_id: (pan, tilt)}
            self._pointing_gains = {}  # {track_id: (k_pan, k_tilt)}
            
            for track_id, track_rows in grouped_by_track.items():
                print(f"[Pointing] Computing track_id={track_id} ({len(track_rows)} detections)")
                
                # === Tilt별 수평 피팅: cx vs pan ===
                by_tilt = defaultdict(list)
                for row in track_rows:
                    by_tilt[round(row['tilt'], 3)].append((row['pan'], row['cx']))
                
                fits_h = {}  # tilt -> dict
                for tkey, arr in by_tilt.items():
                    if len(arr) < min_samples:
                        continue
                    arr.sort(key=lambda v: v[0])
                    pans = np.array([p for p, _ in arr], float)
                    cxs = np.array([c for _, c in arr], float)
                    A = np.vstack([pans, np.ones_like(pans)]).T
                    a, b = np.linalg.lstsq(A, cxs, rcond=None)[0]
                    
                    # R^2
                    yhat = a * pans + b
                    ss_res = float(np.sum((cxs - yhat)**2))
                    ss_tot = float(np.sum((cxs - np.mean(cxs))**2)) + 1e-9
                    R2 = 1.0 - ss_res / ss_tot
                    pan_center = (W_frame / 2.0 - b) / a if abs(a) > 1e-9 else np.nan
                    
                    fits_h[float(tkey)] = {
                        "a": float(a), "b": float(b), "R2": float(R2),
                        "N": int(len(arr)), "pan_center": float(pan_center),
                    }
                
                # === Pan별 수직 피팅: cy vs tilt ===
                by_pan = defaultdict(list)
                for row in track_rows:
                    by_pan[round(row['pan'], 3)].append((row['tilt'], row['cy']))
                
                fits_v = {}  # pan -> dict
                for pkey, arr in by_pan.items():
                    if len(arr) < min_samples:
                        continue
                    arr.sort(key=lambda v: v[0])
                    tilts = np.array([t for t, _ in arr], float)
                    cys = np.array([c for _, c in arr], float)
                    A = np.vstack([tilts, np.ones_like(tilts)]).T
                    e, f = np.linalg.lstsq(A, cys, rcond=None)[0]
                    
                    yhat = e * tilts + f
                    ss_res = float(np.sum((cys - yhat)**2))
                    ss_tot = float(np.sum((cys - np.mean(cys))**2)) + 1e-9
                    R2 = 1.0 - ss_res / ss_tot
                    tilt_center = (H_frame / 2.0 - f) / e if abs(e) > 1e-9 else np.nan
                    
                    fits_v[float(pkey)] = {
                        "e": float(e), "f": float(f), "R2": float(R2),
                        "N": int(len(arr)), "tilt_center": float(tilt_center),
                    }
                
                # === 가중평균 타깃 계산 ===
                def wavg_center(fits: dict, center_key: str):
                    if not fits:
                        return None
                    vals = np.array([fits[k][center_key] for k in fits], float)
                    w = np.array([fits[k]["N"] for k in fits], float)
                    return float(np.sum(vals * w) / np.sum(w))
                
                pan_target = wavg_center(fits_h, "pan_center")
                tilt_target = wavg_center(fits_v, "tilt_center")
                
                # Gain 역산 (deg/px)
                k_pan = CENTERING_GAIN_PAN
                k_tilt = CENTERING_GAIN_TILT
                if fits_h:
                    sum_a_w = sum(d['a'] * d['N'] for d in fits_h.values())
                    sum_w_h = sum(d['N'] for d in fits_h.values())
                    avg_a = sum_a_w / sum_w_h if sum_w_h > 0 else 0.0
                    if abs(avg_a) > 1e-9:
                        k_pan = abs(1.0 / avg_a)
                if fits_v:
                    sum_e_w = sum(d['e'] * d['N'] for d in fits_v.values())
                    sum_w_v = sum(d['N'] for d in fits_v.values())
                    avg_e = sum_e_w / sum_w_v if sum_w_v > 0 else 0.0
                    if abs(avg_e) > 1e-9:
                        k_tilt = abs(1.0 / avg_e)
                
                # Track ID별 결과 저장
                if pan_target is not None and tilt_target is not None:
                    self.computed_targets[track_id] = (round(pan_target, 3), round(tilt_target, 3))
                    self._pointing_gains[track_id] = (k_pan, k_tilt)
                    print(f"[Pointing] track_id={track_id} → pan={pan_target:.3f}°, tilt={tilt_target:.3f}° "
                          f"(H fits: {len(fits_h)}, V fits: {len(fits_v)}, gain: k_p={k_pan:.5f}, k_t={k_tilt:.5f})")
                else:
                    print(f"[Pointing] track_id={track_id} → 계산 실패 (insufficient data)")
            
            # ⭐ 유사 타깃 병합: ±5° 이내 → 같은 ID로 통합
            MERGE_TOL = 5.0  # deg
            merged = self._merge_similar_targets(self.computed_targets, grouped_by_track, MERGE_TOL, W_frame, H_frame, min_samples)
            if merged:
                self.computed_targets = merged['targets']
                self._pointing_gains = merged['gains']

            # UI/스케줄링 편의를 위해 track_id를 1..N으로 재번호
            self._renumber_computed_targets()
            
            # UI 업데이트 (버튼 생성)
            if self.computed_targets:
                print(f"[Pointing] {len(self.computed_targets)} target(s) after merge")
                if hasattr(self, '_create_target_buttons'):
                    self._create_target_buttons(self.computed_targets)
            else:
                print("[Pointing] No targets computed")
                if hasattr(self, '_create_target_buttons'):
                    self._create_target_buttons({})
        
        except Exception as e:
            print(f"[Pointing] 계산 실패: {e}")
            import traceback
            traceback.print_exc()

    def _renumber_computed_targets(self):
        """computed_targets/_pointing_gains의 키를 항상 1..N으로 재번호"""
        if not hasattr(self, "computed_targets") or not self.computed_targets:
            return

        old_ids = sorted(self.computed_targets.keys())
        expected_ids = list(range(1, len(old_ids) + 1))
        if old_ids == expected_ids:
            return

        id_map = {old_id: new_id for new_id, old_id in enumerate(old_ids, start=1)}
        new_targets = {}
        new_gains = {}

        for old_id in old_ids:
            new_id = id_map[old_id]
            new_targets[new_id] = self.computed_targets[old_id]
            gain = self._pointing_gains.get(old_id, (CENTERING_GAIN_PAN, CENTERING_GAIN_TILT))
            new_gains[new_id] = gain

        self.computed_targets = new_targets
        self._pointing_gains = new_gains
        print(f"[Pointing] ID renumbered: {id_map}")
    
    def _merge_similar_targets(self, targets, grouped_by_track, tol, W_frame, H_frame, min_samples):
        """
        ±tol° 이내의 타깃들을 같은 ID로 병합 후 재계산
        """
        if len(targets) <= 1:
            return None
        
        ids = sorted(targets.keys())
        merged_groups = []  # [[id1, id2, ...], ...]
        used = set()
        
        for i, id_a in enumerate(ids):
            if id_a in used:
                continue
            group = [id_a]
            used.add(id_a)
            pan_a, tilt_a = targets[id_a]
            
            for j in range(i + 1, len(ids)):
                id_b = ids[j]
                if id_b in used:
                    continue
                pan_b, tilt_b = targets[id_b]
                if abs(pan_a - pan_b) <= tol and abs(tilt_a - tilt_b) <= tol:
                    group.append(id_b)
                    used.add(id_b)
            
            merged_groups.append(group)
        
        # 병합이 없으면 (모두 단독 그룹) None 반환
        if all(len(g) == 1 for g in merged_groups):
            return None
        
        print(f"[Pointing] 🔗 유사 타깃 병합 (tol=±{tol}°):")
        new_targets = {}
        new_gains = {}
        
        for group in merged_groups:
            rep_id = min(group)  # 가장 낮은 ID가 대표
            
            if len(group) == 1:
                # 단독 → 그대로 유지
                new_targets[rep_id] = targets[rep_id]
                new_gains[rep_id] = self._pointing_gains.get(rep_id, (CENTERING_GAIN_PAN, CENTERING_GAIN_TILT))
                continue
            
            print(f"  IDs {group} → ID {rep_id} 으로 병합")
            
            # 그룹 내 모든 데이터 합치기
            combined_rows = []
            for tid in group:
                if tid in grouped_by_track:
                    combined_rows.extend(grouped_by_track[tid])
            
            # 합친 데이터로 재계산
            result = self._compute_single_target(combined_rows, W_frame, H_frame, min_samples)
            if result:
                new_targets[rep_id] = result['target']
                new_gains[rep_id] = result['gain']
                print(f"  → pan={result['target'][0]:.3f}°, tilt={result['target'][1]:.3f}° "
                      f"({len(combined_rows)} detections)")
            else:
                # 재계산 실패 시 기존 대표 ID 값 유지
                new_targets[rep_id] = targets[rep_id]
                new_gains[rep_id] = self._pointing_gains.get(rep_id, (CENTERING_GAIN_PAN, CENTERING_GAIN_TILT))
        
        return {'targets': new_targets, 'gains': new_gains}
    
    def _compute_single_target(self, rows, W_frame, H_frame, min_samples):
        """단일 그룹의 데이터로 타깃 pan/tilt 계산 (회귀분석)"""
        by_tilt = defaultdict(list)
        for row in rows:
            by_tilt[round(row['tilt'], 3)].append((row['pan'], row['cx']))
        
        fits_h = {}
        for tkey, arr in by_tilt.items():
            if len(arr) < min_samples:
                continue
            arr.sort(key=lambda v: v[0])
            pans = np.array([p for p, _ in arr], float)
            cxs = np.array([c for _, c in arr], float)
            A = np.vstack([pans, np.ones_like(pans)]).T
            a, b = np.linalg.lstsq(A, cxs, rcond=None)[0]
            pan_center = (W_frame / 2.0 - b) / a if abs(a) > 1e-9 else np.nan
            fits_h[float(tkey)] = {"a": float(a), "N": len(arr), "pan_center": float(pan_center)}
        
        by_pan = defaultdict(list)
        for row in rows:
            by_pan[round(row['pan'], 3)].append((row['tilt'], row['cy']))
        
        fits_v = {}
        for pkey, arr in by_pan.items():
            if len(arr) < min_samples:
                continue
            arr.sort(key=lambda v: v[0])
            tilts = np.array([t for t, _ in arr], float)
            cys = np.array([c for _, c in arr], float)
            A = np.vstack([tilts, np.ones_like(tilts)]).T
            e, f = np.linalg.lstsq(A, cys, rcond=None)[0]
            tilt_center = (H_frame / 2.0 - f) / e if abs(e) > 1e-9 else np.nan
            fits_v[float(pkey)] = {"e": float(e), "N": len(arr), "tilt_center": float(tilt_center)}
        
        def wavg(fits, key):
            if not fits: return None
            vals = np.array([fits[k][key] for k in fits], float)
            w = np.array([fits[k]["N"] for k in fits], float)
            return float(np.sum(vals * w) / np.sum(w))
        
        pan_t = wavg(fits_h, "pan_center")
        tilt_t = wavg(fits_v, "tilt_center")
        if pan_t is None or tilt_t is None:
            return None
        
        # gain
        k_pan, k_tilt = CENTERING_GAIN_PAN, CENTERING_GAIN_TILT
        if fits_h:
            avg_a = sum(d['a'] * d['N'] for d in fits_h.values()) / sum(d['N'] for d in fits_h.values())
            if abs(avg_a) > 1e-9: k_pan = abs(1.0 / avg_a)
        if fits_v:
            avg_e = sum(d['e'] * d['N'] for d in fits_v.values()) / sum(d['N'] for d in fits_v.values())
            if abs(avg_e) > 1e-9: k_tilt = abs(1.0 / avg_e)
        
        return {
            'target': (round(pan_t, 3), round(tilt_t, 3)),
            'gain': (k_pan, k_tilt)
        }

    # ========== Laser Fine-Aiming ==========

    def set_pointing_mode(self, mode):
        """Pointing 모드 선택: rough | legacy"""
        mode = str(mode or "").strip().lower()
        if mode not in ("rough", "legacy"):
            print(f"[Pointing] 알 수 없는 모드: {mode} (사용 가능: rough, legacy)")
            return
        self.pointing_mode = mode
        print(f"[Pointing] 모드 변경: {mode}")
    
    def move_to_target(self, track_id, use_tilt_approach=False):
        """
        특정 track_id의 계산된 pan/tilt로 초기 위치 이동만 수행
        """
        if not hasattr(self, 'computed_targets') or track_id not in self.computed_targets:
            print(f"[Pointing] Track {track_id} 타깃 없음. 먼저 계산하세요")
            return
        
        # 이미 조준 중이면 무시
        if hasattr(self, '_aiming_active') and self._aiming_active:
            print(f"[Pointing] 이미 조준 중입니다. 완료될 때까지 대기하세요.")
            return

        pan_t, tilt_t = self.computed_targets[track_id]

        # Start 버튼에서 사용할 현재 선택 타깃 저장
        self._selected_track_id = track_id
        self._curr_pan = pan_t
        self._curr_tilt = tilt_t

        print(f"[Pointing] Track {track_id} 선택. 초기 위치로 이동: pan={pan_t}°, tilt={tilt_t}°")

        # 초기 이동 (필요 시 tilt+1 -> 목표 tilt 접근)
        if use_tilt_approach:
            self._apply_final_tilt_approach(pan_t, tilt_t, settle_s=max(0.05, self.scan_tab.settle.get()))
        else:
            spd = 100
            acc = 1.0
            self.ctrl.send({"cmd": "move", "pan": pan_t, "tilt": tilt_t, "speed": spd, "acc": acc})
        print(f"[Pointing] 초기 이동 명령 전송 완료. Start Aiming 대기 중")

    def start_aiming(self, track_id=None):
        """
        선택된 track_id에 대해 정밀 조준 알고리즘 시작
        """
        if track_id is None:
            track_id = getattr(self, '_selected_track_id', None)
        else:
            self._selected_track_id = track_id

        if track_id is None:
            print("[Pointing] Start 실패: 선택된 타깃이 없습니다")
            return False

        if not hasattr(self, 'computed_targets') or track_id not in self.computed_targets:
            print(f"[Pointing] Start 실패: Track {track_id} 타깃 없음")
            return False

        if hasattr(self, '_aiming_active') and self._aiming_active:
            print("[Pointing] 이미 조준 중입니다. 완료될 때까지 대기하세요.")
            return False

        pan_t, tilt_t = self.computed_targets[track_id]

        # Aiming 시작 전 Preview 상태 저장 (종료/중단 시 복구용)
        self._aiming_restore_preview = bool(getattr(self, 'preview_active', False))
        if self._aiming_restore_preview and hasattr(self, '_get_preview_cfg'):
            self._aiming_preview_cfg = self._get_preview_cfg()
        else:
            self._aiming_preview_cfg = None

        mode = getattr(self, "pointing_mode", "rough")
        if mode not in ("rough", "legacy"):
            mode = "rough"

        print(f"[Pointing] ===== Track {track_id} Fine-Aiming 시작 (mode={mode}) =====")
        print(f"[Pointing] 기준 위치: pan={pan_t}°, tilt={tilt_t}°")

        # IR Mode로 전환 (IR Laser 가시화)
        print("[Pointing] IR Mode로 전환...")
        self.set_ir_cut("day")  # day = IR 필터 해제 → IR 레이저 보임
        time.sleep(0.5)

        self._aiming_active = True
        self._aiming_cancel_event = threading.Event()
        self._aiming_track_id = track_id
        self._curr_pan = pan_t
        self._curr_tilt = tilt_t
        
        # 이미지 수신 대기용 Event + 저장소
        self._pointing_img_event = threading.Event()
        self._pointing_img_data = None
        
        thread_target = self._fine_aim_thread_rough if mode == "rough" else self._fine_aim_thread
        t = threading.Thread(target=thread_target, args=(track_id,), daemon=True)
        t.start()
        return True

    def _restore_preview_after_aiming(self, reason="aiming"):
        """Aiming 종료/중단 후 Preview 복구 (시작 전 ON이었던 경우만)"""
        if not getattr(self, '_aiming_restore_preview', False):
            return

        # 사용자가 중간에 직접 OFF했다면 복구하지 않음
        if not getattr(self, 'preview_active', False):
            self._aiming_restore_preview = False
            self._aiming_preview_cfg = None
            return

        cfg = getattr(self, '_aiming_preview_cfg', None)
        self._aiming_restore_preview = False
        self._aiming_preview_cfg = None

        def _do_restore():
            if hasattr(self, '_restore_preview'):
                self._restore_preview(cfg, reason=reason)
            elif hasattr(self, '_restart_preview'):
                self._restart_preview()

        # Pi snap thread 정리 시간을 조금 준 뒤 복구
        self.root.after(400, _do_restore)
    
    def _snap_and_wait(self, label, timeout=10.0, shutter_speed=None, analogue_gain=None):
        """
        Snap 명령 전송 후 이미지 수신 대기 (Thread-blocking)
        """
        if not getattr(self, '_aiming_active', False):
            return None

        self._pointing_img_event.clear()
        self._pointing_img_data = None
        
        # Scan 해상도 사용 (고해상도)
        w = self.scan_tab.width.get()
        h = self.scan_tab.height.get()
        q = self.scan_tab.quality.get()
        
        cmd = {
            "cmd": "snap",
            "width": w,
            "height": h,
            "quality": q,
            "save": f"{label}.jpg"
        }
        
        # 노출 제어 파라미터 추가
        if shutter_speed is not None:
            cmd["shutter_speed"] = int(shutter_speed)
        if analogue_gain is not None:
            cmd["analogue_gain"] = float(analogue_gain)
            
        self.ctrl.send(cmd)

        # 이미지 수신 대기 (stop_aiming 즉시 반영을 위해 short-poll)
        deadline = time.monotonic() + float(timeout)
        poll_s = 0.1
        while True:
            if not getattr(self, '_aiming_active', False):
                print(f"[Pointing] Snap wait cancelled (inactive): {label}")
                return None

            cancel_evt = getattr(self, '_aiming_cancel_event', None)
            if cancel_evt is not None and cancel_evt.is_set():
                print(f"[Pointing] Snap wait cancelled (event): {label}")
                return None

            remain = deadline - time.monotonic()
            if remain <= 0:
                print(f"[Pointing] ⚠️ Snap timeout: {label}")
                return None

            if self._pointing_img_event.wait(timeout=min(poll_s, remain)):
                # stop 이후 set()된 이벤트일 수 있으므로 한 번 더 체크
                if not getattr(self, '_aiming_active', False):
                    return None
                if cancel_evt is not None and cancel_evt.is_set():
                    return None
                return self._pointing_img_data
    
    def _on_pointing_image_received(self, name, data):
        """
        Pointing용 이미지 수신 콜백 (event_handlers에서 호출)
        """
        try:
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(data))
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            self._pointing_img_data = bgr
            self._pointing_img_event.set()
        except Exception as e:
            print(f"[Pointing] 이미지 디코딩 실패: {e}")
            self._pointing_img_event.set()

    def _apply_final_tilt_approach(self, pan, tilt, settle_s=0.1):
        """최종 수렴 시 tilt +1 -> 원래 tilt 접근 무빙"""
        pan_f = float(pan)
        tilt_f = float(tilt)
        pre_tilt = max(-30.0, min(90.0, tilt_f + FINAL_TILT_APPROACH_UP_DEG))
        wait_s = max(0.02, float(settle_s))

        # 1) 한 칸 위로
        self.ctrl.send({
            "cmd": "move",
            "pan": pan_f,
            "tilt": pre_tilt,
            "speed": 100,
            "acc": 1.0,
        })
        time.sleep(wait_s)

        # 2) 목표 tilt로 하강
        self.ctrl.send({
            "cmd": "move",
            "pan": pan_f,
            "tilt": tilt_f,
            "speed": 100,
            "acc": 1.0,
        })
        time.sleep(wait_s)

    def _fine_aim_thread_rough(self, track_id):
        """
        Rough 조준 Thread:
          Phase 1) YOLO 중심 X를 화면 중심 X에 정렬 (X만 사용)
          Phase 2) 시작 시 tilt +2deg 후, Laser ON/OFF(shutter=100) diff의
                   YOLO bbox 평균 밝기를 비교하며 tilt를 1deg씩 내림.
                   이전 대비 급락 시 조준 성공으로 판정.
        """
        try:
            settle = self.scan_tab.settle.get()
            led_settle = self.scan_tab.led_settle.get()

            gains = self._pointing_gains.get(track_id, (CENTERING_GAIN_PAN, CENTERING_GAIN_TILT))
            k_pan, k_tilt = gains

            tol_phase1_x = ROUGH_PHASE1_TOL_X_PX
            phase = 1
            last_px_per_cm = None
            phase1_pan_hist = []
            phase1_err_hist = []
            phase1_best_pan = float(getattr(self, "_curr_pan", 0.0))
            phase1_best_abs_err = float("inf")
            phase2_prev_mean = None
            phase2_prev_tilt = None

            print(
                f"[Pointing-Rough] 파라미터: settle={settle}s, led_settle={led_settle}s, "
                f"k_pan={k_pan:.5f}, k_tilt={k_tilt:.5f}, "
                f"tol_p1_x={tol_phase1_x}px, phase2_start_up={ROUGH_PHASE2_START_TILT_UP_DEG}deg, "
                f"phase2_step={ROUGH_PHASE2_TILT_STEP_DEG}deg, drop_ratio={ROUGH_PHASE2_DROP_RATIO}, "
                f"drop_delta={ROUGH_PHASE2_DROP_DELTA}"
            )

            time.sleep(max(settle, 1.0))
            self.ctrl.send({"cmd": "laser", "value": 0})
            self._update_aiming_status(track_id, 0, "Rough 시작: YOLO 중심 기반 조준")

            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"Captures/Pointing/{now_str}_Track_{track_id}_rough"
            os.makedirs(log_dir, exist_ok=True)
            print(f"[Pointing-Rough] Logging to: {log_dir}")

            iteration = 0
            while self._aiming_active:
                iteration += 1
                phase_name = "CenterX" if phase == 1 else "Brightness"
                print(f"\n[Pointing-Rough] ===== Iteration {iteration} / Phase {phase_name} =====")
                self._update_aiming_status(track_id, iteration, f"Rough 반복 {iteration} ({phase_name})")

                # Step 1: YOLO 객체 중심 검출(LED diff)
                self.ctrl.send({"cmd": "led", "value": 255})
                time.sleep(led_settle)
                img_led_on = self._snap_and_wait(
                    f"pointing_rough_led_on_{iteration}",
                    shutter_speed=10000,
                    analogue_gain=None,
                )
                if img_led_on is None:
                    self.ctrl.send({"cmd": "led", "value": 0})
                    print("[Pointing-Rough] ⚠️ LED ON 이미지 수신 실패")
                    continue

                self.ctrl.send({"cmd": "led", "value": 0})
                time.sleep(led_settle)
                img_led_off = self._snap_and_wait(
                    f"pointing_rough_led_off_{iteration}",
                    shutter_speed=10000,
                    analogue_gain=None,
                )
                if img_led_off is None:
                    print("[Pointing-Rough] ⚠️ LED OFF 이미지 수신 실패")
                    continue

                try:
                    cv2.imwrite(f"{log_dir}/iter_{iteration}_led_on.jpg", img_led_on)
                    cv2.imwrite(f"{log_dir}/iter_{iteration}_led_off.jpg", img_led_off)
                except Exception as e:
                    print(f"[Pointing-Rough] Log save failed: {e}")

                obj_cx, obj_cy, bbox, all_bboxes = self._find_object_center(img_led_on, img_led_off)
                if obj_cx is None:
                    print("[Pointing-Rough] ⚠️ 객체 검출 실패")
                    continue

                H, W = img_led_on.shape[:2]
                frame_cx = W / 2.0
                frame_cy = H / 2.0

                if bbox and bbox[2] > 0:
                    px_per_cm = float(bbox[2]) / OBJECT_SIZE_CM
                    last_px_per_cm = px_per_cm
                elif last_px_per_cm is not None:
                    px_per_cm = last_px_per_cm
                else:
                    # bbox가 없는 첫 프레임 fallback
                    px_per_cm = 10.0

                if phase == 1:
                    # Phase 1: YOLO 중심 X를 화면 중심 X에 맞춤 (X only)
                    target_x = frame_cx
                    target_y = frame_cy
                    ref_x = obj_cx
                    ref_y = obj_cy
                    err_x = obj_cx - frame_cx

                    print(f"[Pointing-Rough] CenterX: err_x={err_x:.1f}px, px_per_cm={px_per_cm:.3f}")

                    self._draw_debug_image(
                        img_led_on,
                        target_x,
                        target_y,
                        (int(ref_x), int(ref_y)),
                        bbox,
                        all_bboxes,
                        err_x,
                        0.0,
                        iteration,
                        None,
                        None,
                        log_dir,
                        tol_x=tol_phase1_x,
                        tol_y=9999,
                    )

                    self._update_aiming_status(
                        track_id,
                        iteration,
                        f"Rough CenterX: err_x={err_x:.1f}px (tol_x={tol_phase1_x}px)",
                    )

                    # Phase 1 진동(예: 26/27 반복) 감지용 히스토리
                    cur_pan_int = int(self._curr_pan)
                    phase1_pan_hist.append(cur_pan_int)
                    phase1_err_hist.append(float(err_x))
                    if len(phase1_pan_hist) > ROUGH_PHASE1_STUCK_WINDOW:
                        phase1_pan_hist.pop(0)
                        phase1_err_hist.pop(0)

                    cur_abs_err = abs(err_x)
                    if cur_abs_err < phase1_best_abs_err:
                        phase1_best_abs_err = cur_abs_err
                        phase1_best_pan = float(self._curr_pan)

                    phase1_stuck = False
                    if len(phase1_pan_hist) >= ROUGH_PHASE1_STUCK_WINDOW:
                        p = phase1_pan_hist[-ROUGH_PHASE1_STUCK_WINDOW:]
                        is_two_level_repeat = len(set(p)) == 2
                        transitions = sum(1 for i in range(1, len(p)) if p[i] != p[i - 1])
                        returned_to_start = p[0] == p[-1]
                        level_counts_ok = all(p.count(v) >= 2 for v in set(p))
                        all_not_converged = all(abs(e) > tol_phase1_x for e in phase1_err_hist[-ROUGH_PHASE1_STUCK_WINDOW:])
                        phase1_stuck = (
                            is_two_level_repeat
                            and transitions >= 2
                            and returned_to_start
                            and level_counts_ok
                            and all_not_converged
                        )

                    if cur_abs_err <= tol_phase1_x or phase1_stuck:
                        if phase1_stuck:
                            lock_pan = float(int(phase1_best_pan))
                            self._curr_pan = lock_pan
                            self.ctrl.send(
                                {
                                    "cmd": "move",
                                    "pan": lock_pan,
                                    "tilt": self._curr_tilt,
                                    "speed": 100,
                                    "acc": 1.0,
                                }
                            )
                            print(
                                f"[Pointing-Rough] ⚠️ Phase 1 진동 감지(2-level repeat), "
                                f"best pan={lock_pan:.0f}, best |err_x|={phase1_best_abs_err:.1f}px로 고정"
                            )
                            self._update_aiming_status(
                                track_id,
                                iteration,
                                f"Phase1 진동 감지 -> best pan={lock_pan:.0f} 고정 후 Phase2",
                            )
                        phase = 2
                        # Phase 2 시작 전 기본 상향 오프셋(+2deg)
                        self._curr_tilt = self._curr_tilt + ROUGH_PHASE2_START_TILT_UP_DEG
                        self.ctrl.send(
                            {
                                "cmd": "move",
                                "pan": self._curr_pan,
                                "tilt": self._curr_tilt,
                                "speed": 100,
                                "acc": 1.0,
                            }
                        )
                        phase2_prev_mean = None
                        phase2_prev_tilt = None
                        print(
                            "[Pointing-Rough] ✅ Phase 1(X) 수렴 -> Phase 2 시작 "
                            f"(tilt +{ROUGH_PHASE2_START_TILT_UP_DEG}deg)"
                        )
                        self._update_aiming_status(track_id, iteration, "Phase 1(X) 수렴, tilt +2 후 Phase 2 시작")
                        time.sleep(settle)
                        continue

                    d_pan = err_x * k_pan
                    d_pan = max(min(d_pan, MAX_STEP_DEG), -MAX_STEP_DEG)
                    d_tilt = 0.0
                    next_pan = self._curr_pan + d_pan
                    next_tilt = self._curr_tilt
                    next_pan = max(-180, min(180, next_pan))

                    self._curr_pan = next_pan
                    self._curr_tilt = next_tilt

                    self.ctrl.send(
                        {
                            "cmd": "move",
                            "pan": next_pan,
                            "tilt": next_tilt,
                            "speed": 100,
                            "acc": 1.0,
                        }
                    )

                    try:
                        with open(f"{log_dir}/log.txt", "a", encoding="utf-8") as f:
                            f.write(
                                "Iter {it} Phase CenterX: ObjX={ox:.1f} ErrX={ex:.1f} "
                                "d_pan={dp:.3f} Next=({np:.3f},{nt:.3f})\n".format(
                                    it=iteration,
                                    ox=obj_cx,
                                    ex=err_x,
                                    dp=d_pan,
                                    np=next_pan,
                                    nt=next_tilt,
                                )
                            )
                    except Exception as e:
                        print(f"[Pointing-Rough] Log write failed: {e}")

                    time.sleep(settle)
                    continue

                # Phase 2: Laser ON/OFF diff에서 YOLO bbox 영역 평균 밝기 기반
                print("[Pointing-Rough] Phase 2: Laser ON...")
                self.ctrl.send({"cmd": "laser", "value": 1})
                time.sleep(led_settle)
                img_laser_on = self._snap_and_wait(
                    f"pointing_rough_laser_on_{iteration}",
                    shutter_speed=100,
                    analogue_gain=1.0,
                )
                if img_laser_on is None:
                    print("[Pointing-Rough] ⚠️ Laser ON 이미지 수신 실패")
                    continue

                print("[Pointing-Rough] Phase 2: Laser OFF...")
                self.ctrl.send({"cmd": "laser", "value": 0})
                time.sleep(led_settle)
                img_laser_off = self._snap_and_wait(
                    f"pointing_rough_laser_off_{iteration}",
                    shutter_speed=100,
                    analogue_gain=1.0,
                )
                if img_laser_off is None:
                    print("[Pointing-Rough] ⚠️ Laser OFF 이미지 수신 실패")
                    continue

                try:
                    cv2.imwrite(f"{log_dir}/iter_{iteration}_laser_on.jpg", img_laser_on)
                    cv2.imwrite(f"{log_dir}/iter_{iteration}_laser_off.jpg", img_laser_off)
                except Exception as e:
                    print(f"[Pointing-Rough] Log save failed: {e}")

                laser_diff = cv2.absdiff(img_laser_on, img_laser_off)
                laser_gray = cv2.cvtColor(laser_diff, cv2.COLOR_BGR2GRAY)

                if not bbox:
                    print("[Pointing-Rough] ⚠️ Phase 2: YOLO bbox 없음, 밝기 비교 스킵")
                    time.sleep(settle)
                    continue

                bx, by, bw, bh = [int(v) for v in bbox]
                x1 = max(0, bx)
                y1 = max(0, by)
                x2 = min(W, bx + bw)
                y2 = min(H, by + bh)
                if x2 <= x1 or y2 <= y1:
                    print("[Pointing-Rough] ⚠️ Phase 2: YOLO bbox ROI 유효하지 않음")
                    time.sleep(settle)
                    continue

                roi = laser_gray[y1:y2, x1:x2]
                mean_bright = float(np.mean(roi)) if roi.size > 0 else 0.0

                print(
                    f"[Pointing-Rough] Brightness(BBox ROI {x1}:{x2}, {y1}:{y2}) "
                    f"mean={mean_bright:.2f}"
                )

                self._update_aiming_status(
                    track_id,
                    iteration,
                    f"Rough Brightness: mean={mean_bright:.1f} (bbox)",
                )

                # 직전 tilt 대비 밝기 급락이면 현재 tilt(급락 지점)를 조준 성공으로 판정
                if phase2_prev_mean is not None:
                    drop_delta = phase2_prev_mean - mean_bright
                    drop_ratio = mean_bright / max(phase2_prev_mean, 1e-6)
                    is_drop = (drop_ratio <= ROUGH_PHASE2_DROP_RATIO) and (drop_delta >= ROUGH_PHASE2_DROP_DELTA)

                    if is_drop:
                        final_pan = self._curr_pan
                        final_tilt = self._curr_tilt
                        self._curr_pan = final_pan
                        self._curr_tilt = final_tilt
                        self.ctrl.send(
                            {
                                "cmd": "move",
                                "pan": final_pan,
                                "tilt": final_tilt,
                                "speed": 100,
                                "acc": 1.0,
                            }
                        )
                        self.computed_targets[track_id] = (round(final_pan, 3), round(final_tilt, 3))
                        self._update_target_button_value(track_id, round(final_pan, 3), round(final_tilt, 3))
                        self.ctrl.send({"cmd": "laser", "value": 1})
                        print(
                            f"[Pointing-Rough] ✅ Phase 2 완료: 밝기 급락 감지 "
                            f"(prev={phase2_prev_mean:.2f}, cur={mean_bright:.2f}, "
                            f"ratio={drop_ratio:.3f}, delta={drop_delta:.2f})"
                        )
                        self._update_aiming_status(track_id, iteration, "✅ Rough 완료: 밝기 급락 감지")
                        try:
                            with open(f"{log_dir}/log.txt", "a", encoding="utf-8") as f:
                                f.write(
                                    "Iter {it} Phase Brightness-FinalDrop: Prev={pm:.2f} Cur={cm:.2f} "
                                    "Ratio={rr:.3f} Delta={dd:.2f} Final=({fp:.3f},{ft:.3f}) "
                                    "ROI=({x1},{y1},{x2},{y2})\n".format(
                                        it=iteration,
                                        pm=phase2_prev_mean,
                                        cm=mean_bright,
                                        rr=drop_ratio,
                                        dd=drop_delta,
                                        fp=final_pan,
                                        ft=final_tilt,
                                        x1=x1, y1=y1, x2=x2, y2=y2,
                                    )
                                )
                        except Exception as e:
                            print(f"[Pointing-Rough] Log write failed: {e}")
                        break

                # 다음 비교를 위해 현재값 저장 후 tilt를 1도 내려 탐색 계속
                phase2_prev_mean = mean_bright
                phase2_prev_tilt = self._curr_tilt
                next_pan = self._curr_pan
                next_tilt = self._curr_tilt - ROUGH_PHASE2_TILT_STEP_DEG
                self._curr_pan = next_pan
                self._curr_tilt = next_tilt
                self.ctrl.send(
                    {
                        "cmd": "move",
                        "pan": next_pan,
                        "tilt": next_tilt,
                        "speed": 100,
                        "acc": 1.0,
                    }
                )

                try:
                    with open(f"{log_dir}/log.txt", "a", encoding="utf-8") as f:
                        f.write(
                            "Iter {it} Phase Brightness-Search: Mean={mb:.2f} "
                            "NextTilt={nt:.3f} ROI=({x1},{y1},{x2},{y2})\n".format(
                                it=iteration,
                                mb=mean_bright,
                                nt=next_tilt,
                                x1=x1, y1=y1, x2=x2, y2=y2,
                            )
                        )
                except Exception as e:
                    print(f"[Pointing-Rough] Log write failed: {e}")

                time.sleep(settle)

        except Exception as e:
            print(f"[Pointing-Rough] ❌ 오류: {e}")
            import traceback
            traceback.print_exc()
            self._update_aiming_status(track_id, 0, f"❌ 오류: {e}")

        finally:
            self._aiming_active = False
            self._aiming_track_id = None
            self._restore_preview_after_aiming(reason="aiming-end")
            print("[Pointing-Rough] Thread 종료")

    def _fine_aim_thread(self, track_id):
        """
        정밀 조준 Thread (blocking 방식) - 반복 횟수 제한 없음
        """
        try:
            settle = self.scan_tab.settle.get()      # 이동 후 대기 (s)
            led_settle = self.scan_tab.led_settle.get()  # LED/Laser 토글 후 대기 (s)
            
            # Gain (회귀에서 역산된 값 사용)
            gains = self._pointing_gains.get(track_id, (CENTERING_GAIN_PAN, CENTERING_GAIN_TILT))
            k_pan, k_tilt = gains
            
            print(f"[Pointing] Fine-aim 파라미터: settle={settle}s, led_settle={led_settle}s, "
                  f"k_pan={k_pan:.5f}, k_tilt={k_tilt:.5f}")
            
            # 초기 이동 대기
            time.sleep(max(settle, 1.0))
            
            # UI 상태 업데이트
            self._update_aiming_status(track_id, 0, "초기 이동 완료, 조준 시작...")
            
            # [LOG] 세션 로깅 디렉토리 생성
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"Captures/Pointing/{now_str}_Track_{track_id}"
            os.makedirs(log_dir, exist_ok=True)
            print(f"[Pointing] Logging to: {log_dir}")
            
            iteration = 0
            while self._aiming_active:
                iteration += 1
                
                print(f"\n[Pointing] ===== Iteration {iteration} =====")
                self._update_aiming_status(track_id, iteration, f"반복 {iteration}")
                
                # ---- Step 1: Object Center Detection (LED) ----
                # Auto Exposure (shutter=None) 사용
                
                print("[Pointing] Step 1: LED ON...")
                self.ctrl.send({"cmd": "led", "value": 255})
                time.sleep(led_settle)
                
                img_led_on = self._snap_and_wait(f"pointing_led_on_{iteration}", shutter_speed=10000, analogue_gain=None)
                if img_led_on is None:
                    print("[Pointing] ⚠️ LED ON 이미지 수신 실패")
                    self.ctrl.send({"cmd": "led", "value": 0})
                    continue
                
                print("[Pointing] Step 1: LED OFF...")
                self.ctrl.send({"cmd": "led", "value": 0})
                time.sleep(led_settle)
                
                img_led_off = self._snap_and_wait(f"pointing_led_off_{iteration}", shutter_speed=10000, analogue_gain=None)
                if img_led_off is None:
                    print("[Pointing] ⚠️ LED OFF 이미지 수신 실패")
                    continue
                
                # [LOG] LED 이미지 저장
                try:
                    cv2.imwrite(f"{log_dir}/iter_{iteration}_led_on.jpg", img_led_on)
                    cv2.imwrite(f"{log_dir}/iter_{iteration}_led_off.jpg", img_led_off)
                except Exception as e:
                    print(f"[Pointing] Log save failed: {e}")
                
                # ⭐ 타겟 찾기 호출
                target_cx, target_cy, bbox, all_bboxes = self._find_object_center(img_led_on, img_led_off)
                if target_cx is None:
                    print("[Pointing] ⚠️ 객체 검출 실패 → 현재 위치 유지")
                    all_bboxes = []
                    continue
                
                # 타겟 오프셋 적용 (객체 중심 아래 TARGET_OFFSET_CM)
                obj_cx_raw, obj_cy_raw = target_cx, target_cy
                if bbox:
                    bx, by, bw, bh = bbox
                    px_per_cm = bw / OBJECT_SIZE_CM
                    target_cy = target_cy + abs(TARGET_OFFSET_CM) * px_per_cm
                    print(f"[Pointing] Target offset: {TARGET_OFFSET_CM}cm → +{abs(TARGET_OFFSET_CM) * px_per_cm:.1f}px")
                
                print(f"[Pointing] ✅ Object center: ({obj_cx_raw:.1f}, {obj_cy_raw:.1f}) → Target: ({target_cx:.1f}, {target_cy:.1f})")
                
                # ---- Step 2: Laser Center Detection (IR Mode 유지) ----
                # Manual Exposure 사용 (레이저 점만 선명하게)
                
                print("[Pointing] Step 2: Laser ON (Low Exposure)...")
                self.ctrl.send({"cmd": "laser", "value": 1})
                time.sleep(led_settle)
                
                img_laser_on = self._snap_and_wait(f"pointing_laser_on_{iteration}", 
                                                   shutter_speed=2500, analogue_gain=1.0)
                if img_laser_on is None:
                    print("[Pointing] ⚠️ Laser ON 이미지 수신 실패")
                    continue
                
                print("[Pointing] Step 2: Laser OFF (Low Exposure)...")
                self.ctrl.send({"cmd": "laser", "value": 0})
                time.sleep(led_settle)
                
                img_laser_off = self._snap_and_wait(f"pointing_laser_off_{iteration}", 
                                                    shutter_speed=2500, analogue_gain=1.0)
                if img_laser_off is None:
                    print("[Pointing] ⚠️ Laser OFF 이미지 수신 실패")
                    continue

                # [LOG] Laser 이미지 저장
                try:
                    cv2.imwrite(f"{log_dir}/iter_{iteration}_laser_on.jpg", img_laser_on)
                    cv2.imwrite(f"{log_dir}/iter_{iteration}_laser_off.jpg", img_laser_off)
                except Exception as e:
                    print(f"[Pointing] Log save failed: {e}")
                
                # ⭐ 레이저 찾기 호출
                laser_pos = self._find_laser_center(img_laser_on, img_laser_off, exclude_bboxes=all_bboxes)
                if laser_pos is None:
                    print("[Pointing] ⚠️ 레이저 중심 검출 실패 → 이미지 중심 사용")
                    H, W = img_laser_on.shape[:2]
                    laser_pos = (W // 2, H // 2)
                else:
                    print(f"[Pointing] ✅ Laser center: {laser_pos}")
                
                # ---- Step 3: Error & Correction ----
                err_x = target_cx - laser_pos[0]
                err_y = target_cy - laser_pos[1]
                err_mag = (err_x**2 + err_y**2)**0.5
                
                print(f"[Pointing] 오차: err_x={err_x:.1f}px, err_y={err_y:.1f}px, "
                      f"magnitude={err_mag:.1f}px")
                
                # ---- Debug Visualization ----
                self._draw_debug_image(
                    img_led_on, target_cx, target_cy, laser_pos, bbox, 
                    all_bboxes, err_x, err_y, iteration,
                    img_laser_on, img_laser_off, log_dir
                )
                
                self._update_aiming_status(
                    track_id, iteration,
                    f"반복 {iteration}: 오차 {err_mag:.1f}px (tol=({CONVERGENCE_TOL_PX_X}, {CONVERGENCE_TOL_PX_Y})px)"
                )
                
                # 수렴 판정
                if abs(err_x) <= CONVERGENCE_TOL_PX_X and abs(err_y) <= CONVERGENCE_TOL_PX_Y:
                    final_pan = self._curr_pan
                    final_tilt = self._curr_tilt
                    self._curr_pan = final_pan
                    self._curr_tilt = final_tilt

                    # 수렴 위치를 해당 Track의 기준 위치로 저장(다음에 ID 버튼 클릭 시 바로 이동)
                    self.computed_targets[track_id] = (
                        round(final_pan, 3),
                        round(final_tilt, 3),
                    )
                    self._update_target_button_value(
                        track_id,
                        round(final_pan, 3),
                        round(final_tilt, 3),
                    )
                    print(f"[Pointing] 🎉 수렴 완료! err=({err_x:.1f}, {err_y:.1f})px, {iteration}회 반복")
                    self._update_aiming_status(
                        track_id, iteration,
                        f"🎉 수렴 완료! 오차 {err_mag:.1f}px ({iteration}회)"
                    )
                    # 레이저 켜둔 채로 종료
                    self.ctrl.send({"cmd": "laser", "value": 1})
                    break
                
                # 각도 보정
                d_pan, d_tilt = self._calculate_angle_delta(err_x, err_y, k_pan, k_tilt)
                
                next_pan = self._curr_pan + d_pan
                next_tilt = self._curr_tilt + d_tilt
                
                # 하드웨어 제한
                next_pan = max(-180, min(180, next_pan))
                next_tilt = max(-30, min(90, next_tilt))
                
                self._curr_pan = next_pan
                self._curr_tilt = next_tilt
                
                print(f"[Pointing] 보정: d_pan={d_pan:.3f}°, d_tilt={d_tilt:.3f}° → "
                      f"next: pan={next_pan:.3f}°, tilt={next_tilt:.3f}°")
                
                # 이동
                self.ctrl.send({
                    "cmd": "move",
                    "pan": next_pan,
                    "tilt": next_tilt,
                    "speed": 100,
                    "acc": 1.0
                })
                
                # [LOG] 데이터 기록
                try:
                    with open(f"{log_dir}/log.txt", "a", encoding="utf-8") as f:
                        log_line = (f"Iter {iteration}: "
                                    f"Target=({target_cx:.1f}, {target_cy:.1f}), "
                                    f"Laser=({laser_pos[0]:.1f}, {laser_pos[1]:.1f}), "
                                    f"Err=({err_x:.1f}, {err_y:.1f}), Mag={err_mag:.1f}, "
                                    f"Correction=({d_pan:.3f}, {d_tilt:.3f}), "
                                    f"Next=({next_pan:.3f}, {next_tilt:.3f}), "
                                    f"Gain=({k_pan:.5f}, {k_tilt:.5f})\n")
                        f.write(log_line)
                except Exception as e:
                    print(f"[Pointing] Log write failed: {e}")
                
                # 이동 안정화 대기
                time.sleep(settle)
        
        except Exception as e:
            print(f"[Pointing] ❌ Fine-aim 오류: {e}")
            import traceback
            traceback.print_exc()
            self._update_aiming_status(track_id, 0, f"❌ 오류: {e}")
        
        finally:
            self._aiming_active = False
            self._aiming_track_id = None
            self._restore_preview_after_aiming(reason="aiming-end")
            print(f"[Pointing] Fine-aim Thread 종료")
    
    def _update_aiming_status(self, track_id, iteration, message):
        """UI에 조준 상태 업데이트 (thread-safe)"""
        try:
            self.root.after(0, lambda: self.info_label.config(
                text=f"🎯 Track {track_id} [{iteration}]: {message}"
            ))
            if hasattr(self, 'pointing_tab') and hasattr(self.pointing_tab, 'update_aim_status'):
                self.root.after(0, lambda: self.pointing_tab.update_aim_status(
                    track_id, iteration, message
                ))
        except Exception:
            pass

    def _update_target_button_value(self, track_id, pan, tilt):
        """Pointing 탭 ID 옆 좌표 라벨 갱신 (thread-safe)"""
        try:
            if hasattr(self, 'pointing_tab') and hasattr(self.pointing_tab, 'update_target_value'):
                self.root.after(
                    0,
                    lambda tid=track_id, p=pan, t=tilt: self.pointing_tab.update_target_value(tid, p, t),
                )
        except Exception:
            pass
    
    def _draw_debug_image(self, base_img, target_cx, target_cy, laser_pos,
                          best_bbox, all_bboxes, err_x, err_y, iteration,
                          img_laser_on=None, img_laser_off=None, log_dir=None,
                          tol_x=None, tol_y=None):
        """
        디버그 시각화 이미지 생성 → Pointing 탭에 표시 + 로깅
        """
        try:
            debug = base_img.copy()
            H, W = debug.shape[:2]
            err_mag = (err_x**2 + err_y**2)**0.5
            
            # 타겟 위치 (빨간색)
            tx, ty = int(target_cx), int(target_cy)
            cv2.circle(debug, (tx, ty), 12, (0, 0, 255), 3)
            cv2.drawMarker(debug, (tx, ty), (0, 0, 255), cv2.MARKER_CROSS, 50, 3)
            cv2.putText(debug, "TARGET", (tx+15, ty-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 레이저 위치 (초록색)
            lx, ly = int(laser_pos[0]), int(laser_pos[1])
            cv2.circle(debug, (lx, ly), 12, (0, 255, 0), 3)
            cv2.drawMarker(debug, (lx, ly), (0, 255, 0), cv2.MARKER_CROSS, 50, 3)
            cv2.putText(debug, "LASER", (lx+15, ly-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 타겟↔레이저 연결선 (흰색 점선 효과)
            cv2.line(debug, (tx, ty), (lx, ly), (255, 255, 255), 1, cv2.LINE_AA)
            
            # Best 객체 BBox (노란색)
            if best_bbox:
                bx, by, bw, bh = [int(v) for v in best_bbox]
                cv2.rectangle(debug, (bx, by), (bx+bw, by+bh), (0, 255, 255), 3)
                cv2.putText(debug, "OBJECT", (bx, by-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 마스킹된 bbox들 (회색)
            if all_bboxes:
                for (mx, my, mw, mh) in all_bboxes:
                    cv2.rectangle(debug, (int(mx), int(my)), 
                                  (int(mx+mw), int(my+mh)), (128, 128, 128), 2)
            
            if tol_x is None:
                tol_x = CONVERGENCE_TOL_PX_X
            if tol_y is None:
                tol_y = CONVERGENCE_TOL_PX_Y

            # 오차 정보
            converged = abs(err_x) <= tol_x and abs(err_y) <= tol_y
            color = (0, 255, 0) if converged else (0, 0, 255)
            cv2.putText(debug, f"Iter {iteration}  Err: ({err_x:.1f}, {err_y:.1f}) = {err_mag:.1f}px",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(debug, f"Tol: ({tol_x}, {tol_y})px",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            # 현재 Pan/Tilt 표시
            cur_pan = getattr(self, '_curr_pan', 0.0)
            cur_tilt = getattr(self, '_curr_tilt', 0.0)
            cv2.putText(debug, f"Pan: {cur_pan:.2f}  Tilt: {cur_tilt:.2f}",
                        (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)
            
            # 400x400 Crop (타겟 중심)
            crop_half = 200
            y1c = max(0, ty - crop_half)
            y2c = min(H, ty + crop_half)
            x1c = max(0, tx - crop_half)
            x2c = min(W, tx + crop_half)
            crop = debug[y1c:y2c, x1c:x2c]
            
            # UI로 전송 (thread-safe)
            self.root.after(
                0,
                lambda img=crop.copy(), ex=err_x, ey=err_y, em=err_mag, it=iteration, tx=tol_x, ty=tol_y:
                self._show_debug_preview(img, ex, ey, em, it, tx, ty),
            )
            
            # ⭐ Laser Diff 이미지 생성 + UI 전송
            if img_laser_on is not None and img_laser_off is not None:
                # 1. Diff
                laser_diff = cv2.absdiff(img_laser_on, img_laser_off)
                laser_gray = cv2.cvtColor(laser_diff, cv2.COLOR_BGR2GRAY)
                
                # 2. Threshold (계산 로직과 동일하게 50)
                _, laser_mask = cv2.threshold(laser_gray, 50, 255, cv2.THRESH_BINARY)
                
                # 3. Masking (객체 영역 제거)
                if all_bboxes:
                    for (mx, my, mw, mh) in all_bboxes:
                        x1, y1 = int(mx), int(my)
                        x2, y2 = int(mx+mw), int(my+mh)
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(W, x2); y2 = min(H, y2)
                        
                        if x1 < x2 and y1 < y2:
                            laser_mask[y1:y2, x1:x2] = 0
                
                # 4. 시각화 (Binary Mask 위에 마커 그리기)
                laser_vis = cv2.cvtColor(laser_mask, cv2.COLOR_GRAY2BGR)
                
                # 레이저 위치 표시 (초록)
                lx, ly = int(laser_pos[0]), int(laser_pos[1])
                cv2.circle(laser_vis, (lx, ly), 10, (0, 255, 0), 2)
                cv2.drawMarker(laser_vis, (lx, ly), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
                cv2.putText(laser_vis, f"LASER ({lx},{ly})", (lx+15, ly-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 타겟 위치도 표시 (빨간)
                cv2.circle(laser_vis, (tx, ty), 8, (0, 0, 255), 2)
                
                # 마스킹 테두리만 표시 (내부는 이미 지워짐)
                if all_bboxes:
                    for (mx, my, mw, mh) in all_bboxes:
                        cv2.rectangle(laser_vis, (int(mx), int(my)),
                                      (int(mx+mw), int(my+mh)), (0, 0, 128), 1)
                        cv2.putText(laser_vis, "MASKED", (int(mx), int(my)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 128), 1)
                                    
                # Detection ROI 표시 (Cyan) - 실제 검출 영역
                roi_size = 300
                cx_img, cy_img = W // 2, H // 2
                rx1 = max(0, cx_img - roi_size)
                rx2 = min(W, cx_img + roi_size)
                ry1 = max(0, cy_img - roi_size - 100)
                ry2 = min(H, cy_img + roi_size)
                
                cv2.rectangle(laser_vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
                cv2.putText(laser_vis, "ROI AREA", (rx1, ry1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                    
                # Wide Crop (1200x900) -> 400x300 리사이즈 (Zoom Out 효과)
                H_l, W_l = laser_vis.shape[:2]
                y1l = max(0, ty - 450)
                y2l = min(H_l, ty + 450)
                x1l = max(0, tx - 600)
                x2l = min(W_l, tx + 600)
                laser_crop = laser_vis[y1l:y2l, x1l:x2l]
                self.root.after(0, lambda img=laser_crop.copy(): self._show_laser_diff(img))
            
            # [LOG] 디버그 이미지 저장
            if log_dir:
                try:
                    cv2.imwrite(f"{log_dir}/iter_{iteration}_debug.jpg", debug)
                except Exception as e:
                    print(f"[Pointing] Debug image save failed: {e}")
            
        except Exception as e:
            print(f"[Pointing] Debug 이미지 생성 오류: {e}")
    
    def _show_debug_preview(self, img_bgr, err_x=0, err_y=0, err_mag=0, iteration=0,
                            tol_x=None, tol_y=None):
        """디버그 프리뷰 이미지를 Pointing 탭에 표시 (main thread)"""
        try:
            from PIL import Image, ImageTk
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb).resize((400, 400), Image.LANCZOS)
            photo = ImageTk.PhotoImage(im)
            
            if hasattr(self, 'pointing_tab') and hasattr(self.pointing_tab, 'debug_preview_label'):
                self.pointing_tab.debug_preview_label.config(image=photo)
                self.pointing_tab.debug_preview_label.image = photo
            
            # 오차 텍스트 업데이트
            if hasattr(self, 'pointing_tab') and hasattr(self.pointing_tab, 'debug_error_label'):
                if tol_x is None:
                    tol_x = CONVERGENCE_TOL_PX_X
                if tol_y is None:
                    tol_y = CONVERGENCE_TOL_PX_Y
                color = "green" if abs(err_x) <= tol_x and abs(err_y) <= tol_y else "red"
                self.pointing_tab.debug_error_label.config(
                    text=f"[Iter {iteration}]  err_x={err_x:.1f}px  err_y={err_y:.1f}px  |err|={err_mag:.1f}px  (tol=({tol_x}, {tol_y})px)",
                    fg=color
                )
        except Exception as e:
            print(f"[Pointing] Debug preview 오류: {e}")
    
    def _show_laser_diff(self, img_bgr):
        """Laser diff 이미지를 Pointing 탭에 표시 (main thread)"""
        try:
            from PIL import Image, ImageTk
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb).resize((400, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(im)
            
            if hasattr(self, 'pointing_tab') and hasattr(self.pointing_tab, 'laser_diff_label'):
                self.pointing_tab.laser_diff_label.config(image=photo)
                self.pointing_tab.laser_diff_label.image = photo
        except Exception as e:
            print(f"[Pointing] Laser diff preview 오류: {e}")
    
    def stop_aiming(self):
        """조준 중단"""
        if hasattr(self, '_aiming_active') and self._aiming_active:
            self._aiming_active = False
            if hasattr(self, '_aiming_cancel_event'):
                self._aiming_cancel_event.set()
            if hasattr(self, '_pointing_img_event'):
                # _snap_and_wait 즉시 해제
                self._pointing_img_event.set()
            self.ctrl.send({"cmd": "laser", "value": 0})
            self.ctrl.send({"cmd": "led", "value": 0})
            print("[Pointing] 조준 중단됨")

    # ========== Helper Functions (Missing Re-added) ==========

    def _find_object_center(self, img_led_on, img_led_off):
        """타겟(반사판) 중심 찾기 - Scan과 동일하게 diff 기반"""
        if img_led_on is None or img_led_off is None:
            return None, None, None, None

        # 1) Scan과 동일: LED ON/OFF 차분(diff)
        diff = cv2.absdiff(img_led_on, img_led_off)

        # 2) YOLO Detection은 diff에 대해 수행
        results = []
        if hasattr(self, 'yolo') and self.yolo:
            # 모델 로드 (없으면 로드 시도)
            if not self.yolo._cached_model:
                if hasattr(self, 'scan_tab') and hasattr(self.scan_tab, 'yolo_weights'):
                    model_path = self.scan_tab.yolo_weights.get()
                    if model_path:
                        self.yolo.get_model(model_path)

            # Scan 쪽과 최대한 동일한 파라미터
            results = self.yolo.detect(diff, conf=0.20, iou=0.45)
        else:
            results = []

        all_bboxes = []
        target_bbox = None
        target_center = None

        H, W = diff.shape[:2]
        center_x, center_y = W // 2, H // 2
        min_dist = float('inf')

        # 3) conf>=0.5 후보가 있으면 우선 사용 (없으면 전체 사용)
        use_results = [r for r in results if len(r) >= 6 and float(r[4]) >= 0.5] or results

        # 4) 가장 중앙에 가까운 객체 선택 (기존 로직 유지)
        for r in use_results:
            x1, y1, x2, y2, conf, cls_id = r
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2

            all_bboxes.append((int(x1), int(y1), int(w), int(h)))

            dist = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                target_bbox = (int(x1), int(y1), int(w), int(h))
                target_center = (cx, cy)

        # 5) YOLO 실패 시: diff에서 blob 기반 fallback (현장 안정성)
        if target_center is None:
            try:
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                # 너무 약하면 20~50 사이로 조정 가능
                _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
                mask = cv2.medianBlur(mask, 5)

                fc = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = fc[0] if len(fc) == 2 else fc[1]

                if cnts:
                    c = max(cnts, key=cv2.contourArea)
                    if cv2.contourArea(c) > 30:  # 작은 노이즈 제거
                        x, y, w, h = cv2.boundingRect(c)
                        cx = x + w / 2.0
                        cy = y + h / 2.0
                        all_bboxes = [(int(x), int(y), int(w), int(h))]
                        return cx, cy, (int(x), int(y), int(w), int(h)), all_bboxes
            except Exception:
                pass

            print("[Pointing] ⚠️ YOLO(diff): 타겟 검출 실패")
            return None, None, None, all_bboxes

        return target_center[0], target_center[1], target_bbox, all_bboxes

    def _find_laser_center(self, img_on, img_off, exclude_bboxes=None):
        """레이저 중심 찾기"""
        if img_on is None or img_off is None:
            return None
            
        H, W = img_on.shape[:2]
        
        # ROI 설정
        crop_size = 300
        cy, cx = H // 2, W // 2
        roi_y1 = max(0, cy - crop_size - 100) 
        roi_y2 = min(H, cy + crop_size - 100)
        roi_x1 = max(0, cx - crop_size)
        roi_x2 = min(W, cx + crop_size)
        
        roi_on = img_on[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_off = img_off[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Diff & band-pass threshold (100~200 구간만 사용)
        diff = cv2.absdiff(roi_on, roi_off)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        thr_low = 100.0
        thr_high = 150.0
        weight = gray.astype(np.float32)
        weight[(weight < thr_low) | (weight > thr_high)] = 0.0
        
        # 마스킹 (객체 영역 제거)
        if exclude_bboxes:
            for (bx, by, bw, bh) in exclude_bboxes:
                rx1 = max(0, bx - roi_x1)
                ry1 = max(0, by - roi_y1)
                rx2 = min(roi_x2 - roi_x1, bx + bw - roi_x1)
                ry2 = min(roi_y2 - roi_y1, by + bh - roi_y1)
                
                if rx1 < rx2 and ry1 < ry2:
                    weight[ry1:ry2, rx1:rx2] = 0.0

        # 밝기 가중 무게중심 (Intensity-weighted centroid)
        ys, xs = np.nonzero(weight > 0)
        if len(xs) == 0:
            return None

        w = weight[ys, xs].astype(np.float64)
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            return None

        roi_cx = int(np.sum(xs * w) / w_sum)
        roi_cy = int(np.sum(ys * w) / w_sum)
        
        return (roi_cx + roi_x1, roi_cy + roi_y1)

    def _calculate_angle_delta(self, err_x, err_y, k_pan, k_tilt):
        """각도 변환 헬퍼 (Com_650nm 방식: tilt 부호 반전 + max step 클램핑)"""
        d_pan = err_x * k_pan
        d_tilt = -err_y * k_tilt
        d_pan = max(min(d_pan, MAX_STEP_DEG), -MAX_STEP_DEG)
        d_tilt = max(min(d_tilt, MAX_STEP_DEG), -MAX_STEP_DEG)
        return d_pan, d_tilt
