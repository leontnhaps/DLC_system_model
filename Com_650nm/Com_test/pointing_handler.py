#!/usr/bin/env python3
"""
Pointing mode handler mixin
Handles all pointing-related logic and CSV analysis
"""

import time
import threading
import cv2
import numpy as np
import pathlib
from tkinter import filedialog
from datetime import datetime
from network import ui_q


class PointingHandlerMixin:
    """Pointing mode logic - laser detection, object tracking, CSV analysis"""
    
    # ========== Laser Center Detection ==========
    
    def _find_laser_center(self, img_on, img_off):
        """
        Find laser center using brightness centroid from diff image.
        No ROI, no Contour, just moments of diff grayscale.
        """
        # ROI: 중앙 ±roi_size (가로) + 위로 200px 확장 (세로)
        # roi_size=200 → 400x600, roi_size=300 → 600x800
        H, W = img_on.shape[:2]
        cx, cy = W // 2, H // 2
        roi_size = self.pointing_roi_size.get()
        
        # 가로: cx ± roi_size
        x1 = max(0, cx - roi_size)
        x2 = min(W, cx + roi_size)
        
        # 세로: (cy - roi_size - 200) ~ (cy + roi_size)
        y1 = max(0, cy - roi_size - 200)  # 위로 200 확장
        y2 = min(H, cy)
        
        roi_on = img_on[y1:y2, x1:x2]
        roi_off = img_off[y1:y2, x1:x2]
        
        # Calculate difference image
        diff = cv2.absdiff(roi_on, roi_off)
        
        # Convert to grayscale
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        cv_thresh = 70
        _, filted_gray = cv2.threshold(gray, cv_thresh, 255, cv2.THRESH_TOZERO)

        # Calculate brightness centroid using moments
        M = cv2.moments(filted_gray)
        if M["m00"] == 0:
            return None
        
        # ROI 내부 좌표
        roi_cx = int(M["m10"] / M["m00"])
        roi_cy = int(M["m01"] / M["m00"])
        
        # 전체 이미지 좌표로 변환 (중요!)
        global_cx = roi_cx + x1
        global_cy = roi_cy + y1
        
        return (global_cx, global_cy)

    # ==== Pointing Mode Logic ====
    def _start_pointing_cycle(self):
        # 1. Laser ON
        self._pointing_state = 1 # WAIT_LASER_ON
        self.ctrl.send({"cmd":"laser", "value":1})
        wait_ms = int(self.point_settle.get() * 1000)
        self.root.after(wait_ms, lambda: self.ctrl.send({
            "cmd":"snap", "width":self.width.get(), "height":self.height.get(),
            "quality":self.quality.get(), "save":"pointing_laser_on.jpg"
        }))

    def _run_pointing_laser_logic(self, img_on, img_off):
        try:
            img_on, img_off = self._undistort_pair(img_on, img_off)
            
            laser_pos = self._find_laser_center(img_on, img_off)
            
            if laser_pos is None:
                self._laser_px = None
                ui_q.put(("toast", "⚠️ Laser not found -> Original Scheme "))
                ui_q.put(("pointing_step_2", None))
                return

            # Laser Found -> Proceed to Object Detection
            self._laser_px = laser_pos
            ui_q.put(("toast", f"✅ Laser Found: {laser_pos}"))
            
            # [DEBUG] Save laser visualization (UD applied!)
            diff_laser = cv2.absdiff(img_on, img_off)  # img_on, img_off는 이미 UD 적용됨!
            debug_laser = cv2.cvtColor(diff_laser, cv2.COLOR_BGR2RGB) if len(diff_laser.shape) == 3 else cv2.cvtColor(diff_laser, cv2.COLOR_GRAY2BGR)
            cv2.circle(debug_laser, laser_pos, 10, (0, 255, 0), 3)  # 녹색 원
            cv2.drawMarker(debug_laser, laser_pos, (0, 255, 0), cv2.MARKER_CROSS, 40, 3)  # 십자 마커
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초 포함
            outdir = pathlib.Path(self.outdir.get())
            debug_path = outdir / f"debug_laser_ud_{ts}.jpg"
            cv2.imwrite(str(debug_path), debug_laser)
            print(f"[DEBUG] Laser saved (UD): {debug_path}, pos={laser_pos}")
            # Trigger LED ON
            ui_q.put(("pointing_step_2", None))
            
        except Exception as e:
            ui_q.put(("toast", f"❌ Pointing Laser Error: {e}"))
            self._pointing_state = 0

    def _run_pointing_object_logic(self, img_on, img_off):
        try:
            img_on, img_off = self._undistort_pair(img_on, img_off)
            
            diff = cv2.absdiff(img_on, img_off)
            
            model = self._get_yolo_model()
            if model is None:
                ui_q.put(("toast", "❌ YOLO 없음"))
                self._pointing_state = 0; return

            device = self._get_device()
            
            from yolo_utils import predict_with_tiling, non_max_suppression
            # YOLO constants
            YOLO_TILE_ROWS = 2
            YOLO_TILE_COLS = 3
            YOLO_TILE_OVERLAP = 0.15
            YOLO_CONF_THRESHOLD = 0.50
            YOLO_IOU_THRESHOLD = 0.45
            
            boxes, scores, classes = predict_with_tiling(model, diff, rows=YOLO_TILE_ROWS, cols=YOLO_TILE_COLS, overlap=YOLO_TILE_OVERLAP, conf=YOLO_CONF_THRESHOLD, iou=YOLO_IOU_THRESHOLD, device=device)
            
            if not boxes:
                ui_q.put(("toast", "⚠️ Object not found -> Retry"))
                self._pointing_state = 0; return # Retry next cycle

            # Find closest to center
            H, W = diff.shape[:2]
            cx, cy = W/2, H/2
            best_idx = -1; min_dist = 999999
            
            for i, (x, y, w, h) in enumerate(boxes):
                obj_cx = x + w/2; obj_cy = y + h/2
                dist = (obj_cx - cx)**2 + (obj_cy - cy)**2
                if dist < min_dist:
                    min_dist = dist; best_idx = i
            
            x, y, w, h = boxes[best_idx]
            obj_cx = x + w/2; obj_cy = y + h/2
            
            # Target Calculation (5cm below center)
            # Assume object is 5cm x 5cm
            px_per_cm = w / 5.0
            target_y_offset = 5.0 * px_per_cm
            target_px = (obj_cx, obj_cy + target_y_offset)
            
            if self._laser_px is not None:
                # [레이저 인식] 
                ref_point = self._laser_px
            else : 
                ref_point = (W/2.0, H/2.0)
            
            err_x = target_px[0] - ref_point[0]
            err_y = target_px[1] - ref_point[1]
            # [DEBUG] Save target visualization (UD applied!)
            debug_target = diff.copy()  # diff는 이미 UD 적용된 img_on, img_off의 차분!
            debug_target = cv2.cvtColor(debug_target, cv2.COLOR_GRAY2BGR) if len(debug_target.shape) == 2 else debug_target
            # 타겟 위치 (빨간색)
            cv2.circle(debug_target, (int(target_px[0]), int(target_px[1])), 12, (0, 0, 255), 3)
            cv2.drawMarker(debug_target, (int(target_px[0]), int(target_px[1])), (0, 0, 255), cv2.MARKER_CROSS, 50, 3)
            cv2.putText(debug_target, "TARGET", (int(target_px[0])+15, int(target_px[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # 레이저 위치 (녹색)
            if self._laser_px is not None:
                cv2.circle(debug_target, self._laser_px, 12, (0, 255, 0), 3)
                cv2.drawMarker(debug_target, self._laser_px, (0, 255, 0), cv2.MARKER_CROSS, 50, 3)
                cv2.putText(debug_target, "LASER", (self._laser_px[0]+15, self._laser_px[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # 객체 BBox (노란색)
            cv2.rectangle(debug_target, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 255), 3)
            cv2.putText(debug_target, "OBJECT", (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # 오차 표시
            cv2.putText(debug_target, f"Err: ({err_x:.1f}, {err_y:.1f})", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초 포함
            outdir = pathlib.Path(self.outdir.get())
            debug_path = outdir / f"debug_target_ud_{ts}.jpg"
            cv2.imwrite(str(debug_path), debug_target)
            print(f"[DEBUG] Target saved (UD): {debug_path}, L={self._laser_px}, T={target_px}")
            
            # Update debug preview in GUI (crop to 400x400 around TARGET)
            H_dbg, W_dbg = debug_target.shape[:2]
            cx_dbg, cy_dbg = int(target_px[0]), int(target_px[1])  # 타겟 중심 기준!
            crop_size = 200  # 400x400 total
            y1_crop = max(0, cy_dbg - crop_size)
            y2_crop = min(H_dbg, cy_dbg + crop_size)
            x1_crop = max(0, cx_dbg - crop_size)
            x2_crop = min(W_dbg, cx_dbg + crop_size)
            debug_crop = debug_target[y1_crop:y2_crop, x1_crop:x2_crop]
            ui_q.put(("debug_preview", debug_crop))
            
            ui_q.put(("toast", f"Err:({err_x:.1f}, {err_y:.1f}) L:{self._laser_px} T:{target_px}"))
            
            # Convergence
            tol = self.pointing_px_tol.get()
            if abs(err_x) <= tol and abs(err_y) <= tol:
                self._pointing_stable_cnt += 1
                ui_q.put(("toast", f"✅ Pointing Converging... {self._pointing_stable_cnt}/{self.pointing_min_frames.get()}"))
            
            else:
                self._pointing_stable_cnt = 0
                
            if self._pointing_stable_cnt >= self.pointing_min_frames.get():
                if self._laser_px is not None:
                    ui_q.put(("toast", "🎉 Pointing Complete!"))
                    self.pointing_enable.set(False); ui_q.put(("preview_on", None))
                    self.ctrl.send({"cmd":"laser", "value":0}); self.laser_on.set(False)
                    self._pointing_state = 0
                    return
                else:
                    ui_q.put(("toast", "⚠️ Center Locked but No Laser -> Scanning Down 1°..."))
                    next_tilt = self._curr_tilt - 1.0 
                    next_tilt = max(-30, min(90, next_tilt))
                    self._curr_tilt = next_tilt
                    # Pan은 그대로 둠
                    self.ctrl.send({"cmd":"move", "pan":self._curr_pan, "tilt":next_tilt, "speed":self.speed.get(), "acc":float(self.acc.get())})
                    
                    # ★ 중요: 움직였으니까 다시 흔들림. 카운트 리셋해서 다시 확인하게 함.
                    self._pointing_stable_cnt = 0
                    
                    # 루프 종료 (다음 사이클에서 레이저 다시 찾아봄)
                    self._pointing_state = 0 
                    self._pointing_last_ts = time.time() * 1000
                    return
                
            # [MOD] 역산된 gain 사용 (없으면 기본값 사용됨)
            k_p = getattr(self, '_computed_gain_pan', None)
            k_t = getattr(self, '_computed_gain_tilt', None)
            
            kwargs = {}
            if k_p is not None: kwargs['k_pan'] = k_p
            if k_t is not None: kwargs['k_tilt'] = k_t
            
            # ▼▼▼ [여기!] 오차가 10 이하면 1도로 제한 거는 코드 추가 ▼▼▼
            if abs(err_x) <= 10.0 and abs(err_y) <= 10.0:
                kwargs['force_max_step'] = 1.0  # 강제로 1도 제한
                # ui_q.put(("toast", "🤏 미세 조정 모드 (Max 1.0°)"))
            
            d_pan, d_tilt = self._calculate_angle_delta(err_x, err_y, **kwargs)
            
            next_pan = self._curr_pan + d_pan
            next_tilt = self._curr_tilt + d_tilt
            
            # Hardware limits
            next_pan = max(-180, min(180, next_pan))
            next_tilt = max(-30, min(90, next_tilt))
            
            self._curr_pan = next_pan
            self._curr_tilt = next_tilt
            
            self.ctrl.send({"cmd":"move", "pan":next_pan, "tilt":next_tilt, "speed":self.speed.get(), "acc":float(self.acc.get())})
            ui_q.put(("toast", f" next pan : {next_pan} next tilt : {next_tilt}"))
            self._pointing_state = 0 # Cycle Done
            self._pointing_last_ts = time.time() * 1000

        except Exception as e:
            ui_q.put(("toast", f"❌ Pointing Object Error: {e}"))
            self._pointing_state = 0

    def on_pointing_toggle(self):
        """Handle pointing mode toggle ON/OFF"""
        if self.pointing_enable.get():
            self.pv_monitor.clear_history()
            ui_q.put(("preview_on", None))
            # Laser OFF when starting
            self.ctrl.send({"cmd":"laser", "value": 0})
            # ==== 좌표 로깅 시작 ====
            try:
                import csv, os
                log_dir = pathlib.Path(self.outdir.get())
                os.makedirs(log_dir, exist_ok=True)
                fname = f"point_xy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                path  = log_dir / fname
                # 열려있던 거 있으면 닫기
                if self._pointing_log_fp:
                    try: self._pointing_log_fp.close()
                    except: pass
                self._pointing_log_fp = open(path, "w", newline="", encoding="utf-8")
                self._pointing_log_writer = csv.writer(self._pointing_log_fp)
                self._pointing_log_writer.writerow(
                    ["ts","pan_cmd_deg","tilt_cmd_deg","mean_cx","mean_cy","err_x_px","err_y_px","W","H","n_dets"]
                )
                self._pointing_logging = True
                ui_q.put(("toast", f"[Point] logging → {path}"))
            except Exception as e:
                self._pointing_logging =False
                ui_q.put(("toast", f"[Point] 로그 시작 실패: {e}"))
            
            # PV 모니터링 자동 시작 제거됨 (아두이노)
            # if hasattr(self, 'pv_monitor') and not self.pv_monitoring.get():
            #     self.start_pv_monitoring()
        else:
            self.laser_on.set(False)
            # CSV 종료
            if self._pointing_log_fp:
                try:
                    self._pointing_log_fp.close()
                    self._pointing_log_fp = None
                    self._pointing_log_writer = None
                    self._pointing_logging = False
                    ui_q.put(("toast", "📄 Pointing log 종료"))
                except Exception as e:
                    ui_q.put(("toast", f"❌ log 종료 실패: {e}"))
            
            # PV 모니터링 자동 중지 제거됨 (아두이노)
            # if hasattr(self, 'pv_monitor') and self.pv_monitoring.get():
            #     self.stop_pv_monitoring()

    def pointing_choose_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV","*.csv")])
        if path:
            self.point_csv_path.set(path)
            self.pointing_compute()

    def pointing_compute(self):
        """
        CSV를 읽어:
          1) tilt별 cx= a*pan + b → pan_center = (W/2 - b)/a
          2) pan별  cy= e*tilt+ f → tilt_center= (H/2 - f)/e
        를 구하고, 각 bin의 샘플 수 N으로 가중평균하여 최종 타깃 pan/tilt 계산.
        """
        path = self.point_csv_path.get().strip()
        if not path:
            ui_q.put(("toast", "CSV를 선택하세요."))
            return

        try:
            import numpy as np, csv
            rows = []
            W_frame = H_frame = None
            conf_min = float(self.point_conf_min.get())
            min_samples = int(self.point_min_samples.get())

            with open(path, newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for d in r:
                    if d.get("conf","")=="":
                        continue
                    conf = float(d["conf"])
                    if conf < conf_min:
                        continue
                    pan  = d.get("pan_deg"); tilt = d.get("tilt_deg")
                    if pan in ("",None) or tilt in ("",None):
                        continue
                    pan = float(pan); tilt = float(tilt)
                    cx = float(d["cx"]); cy = float(d["cy"])
                    W  = int(d["W"]) if d.get("W") else None
                    H  = int(d["H"]) if d.get("H") else None
                    
                    # ⭐ track_id 파싱 (기본값 0)
                    track_id = int(d.get("track_id", 0))
                    
                    if W_frame is None and W: W_frame = W
                    if H_frame is None and H: H_frame = H
                    
                    # ⭐ track_id 포함하여 저장 (튜플 → 딕셔너리)
                    rows.append({
                        'track_id': track_id,
                        'pan': pan,
                        'tilt': tilt,
                        'cx': cx,
                        'cy': cy
                    })

            if not rows:
                ui_q.put(("toast", "CSV에서 조건을 만족하는 행이 없습니다. conf/min_samples 확인."))
                return
            if W_frame is None or H_frame is None:
                ui_q.put(("toast", "CSV에 W/H 정보가 없습니다. (W,H 열 필요)"))
                return

            # ⭐⭐⭐ track_id별로 그룹화 ⭐⭐⭐
            from collections import defaultdict
            grouped_by_track = defaultdict(list)
            for row in rows:
                grouped_by_track[row['track_id']].append(row)
            
            print(f"[Pointing] Found {len(grouped_by_track)} track(s): {list(grouped_by_track.keys())}")
            
            # ⭐ 각 track_id별로 독립적으로 계산
            self.computed_targets = {}  # {track_id: (pan, tilt)}
            
            for track_id, track_rows in grouped_by_track.items():
                print(f"[Pointing] Computing track_id={track_id} ({len(track_rows)} detections)")
                
                # --- tilt별 수평 피팅: cx vs pan
                # ---- tilt별: cx = a*pan + b → pan_center = (W/2 - b)/a
                by_tilt = defaultdict(list)
                for row in track_rows:
                    by_tilt[round(row['tilt'], 3)].append((row['pan'], row['cx']))

                fits_h = {}  # tilt -> dict
                for tkey, arr in by_tilt.items():
                    if len(arr) < min_samples: 
                        continue
                    arr.sort(key=lambda v: v[0])
                    pans = np.array([p for p,_ in arr], float)
                    cxs  = np.array([c for _,c in arr], float)
                    A = np.vstack([pans, np.ones_like(pans)]).T
                    a, b = np.linalg.lstsq(A, cxs, rcond=None)[0]
                    # R^2
                    yhat = a*pans + b
                    ss_res = float(np.sum((cxs - yhat)**2))
                    ss_tot = float(np.sum((cxs - np.mean(cxs))**2)) + 1e-9
                    R2 = 1.0 - ss_res/ss_tot
                    pan_center = (W_frame/2.0 - b)/a if abs(a) > 1e-9 else np.nan
                    fits_h[float(tkey)] = {
                        "a": float(a), "b": float(b), "R2": float(R2),
                        "N": int(len(arr)), "pan_center": float(pan_center),
                    }

                # ---- pan별: cy = e*tilt + f → tilt_center = (H/2 - f)/e
                by_pan = defaultdict(list)
                for row in track_rows:
                    by_pan[round(row['pan'], 3)].append((row['tilt'], row['cy']))

                fits_v = {}  # pan -> dict
                for pkey, arr in by_pan.items():
                    if len(arr) < min_samples:
                        continue
                    arr.sort(key=lambda v: v[0])
                    tilts = np.array([t for t,_ in arr], float)
                    cys   = np.array([c for _,c in arr], float)
                    A = np.vstack([tilts, np.ones_like(tilts)]).T
                    e, f = np.linalg.lstsq(A, cys, rcond=None)[0]
                    yhat = e*tilts + f
                    ss_res = float(np.sum((cys - yhat)**2))
                    ss_tot = float(np.sum((cys - np.mean(cys))**2)) + 1e-9
                    R2 = 1.0 - ss_res/ss_tot
                    tilt_center = (H_frame/2.0 - f)/e if abs(e) > 1e-9 else np.nan
                    fits_v[float(pkey)] = {
                        "e": float(e), "f": float(f), "R2": float(R2),
                        "N": int(len(arr)), "tilt_center": float(tilt_center),
                    }

                # ---- 가중평균 타깃 계산
                def wavg_center(fits: dict, center_key: str):
                    if not fits: return None
                    vals = np.array([fits[k][center_key] for k in fits], float)
                    w    = np.array([fits[k]["N"]          for k in fits], float)
                    return float(np.sum(vals*w)/np.sum(w))

                pan_target  = wavg_center(fits_h, "pan_center")
                tilt_target = wavg_center(fits_v, "tilt_center")
                
                # ⭐ track_id별 결과 저장
                if pan_target is not None and tilt_target is not None:
                    self.computed_targets[track_id] = (round(pan_target, 3), round(tilt_target, 3))
                    print(f"[Pointing] track_id={track_id} → pan={pan_target:.3f}°, tilt={tilt_target:.3f}° (H fits: {len(fits_h)}, V fits: {len(fits_v)})")
                else:
                    print(f"[Pointing] track_id={track_id} → 계산 실패 (insufficient data)")
            
            # ⭐ 전역 저장 (마지막 track의 fits, 센터링/보간에서 사용)
            self._fits_h = fits_h
            self._fits_v = fits_v

            # ---- [NEW] 역산값 (Gain) 계산: 1 / mean_slope (px/deg)
            # 가중 평균 slope 계산 (모든 track 통합)
            sum_a_w = sum(d['a'] * d['N'] for d in fits_h.values())
            sum_w_h = sum(d['N'] for d in fits_h.values())
            avg_a = sum_a_w / sum_w_h if sum_w_h > 0 else 0.0

            sum_e_w = sum(d['e'] * d['N'] for d in fits_v.values())
            sum_w_v = sum(d['N'] for d in fits_v.values())
            avg_e = sum_e_w / sum_w_v if sum_w_v > 0 else 0.0

            # Slope(px/deg) 역수 -> deg/px
            if abs(avg_a) > 1e-9:
                self._computed_gain_pan = abs(1.0 / avg_a)
            else:
                self._computed_gain_pan = None

            if abs(avg_e) > 1e-9:
                self._computed_gain_tilt = abs(1.0 / avg_e)
            else:
                self._computed_gain_tilt = None

            ui_q.put(("toast", f"[Gain 역산] P: {self._computed_gain_pan}, T: {self._computed_gain_tilt}"))

            # ---- 기존 UI 업데이트 (첫 번째 track 사용)
            if self.computed_targets:
                first_track_id = list(self.computed_targets.keys())[0]
                first_pan, first_tilt = self.computed_targets[first_track_id]
                self.point_pan_target.set(first_pan)
                self.point_tilt_target.set(first_tilt)
                
                result_text = f"Found {len(self.computed_targets)} object(s)\\n"
                for tid, (p, t) in self.computed_targets.items():
                    result_text += f"Track {tid}: Pan={p}°, Tilt={t}°\\n"
                self.point_result_lbl.config(text=result_text)
                
                ui_q.put(("toast",
                    f"[Pointing] {len(self.computed_targets)} object(s) computed"))
                
                # ⭐ UI 버튼 업데이트
                if hasattr(self, '_create_target_buttons'):
                    self._create_target_buttons(self.computed_targets)
            else:
                ui_q.put(("toast", "[Pointing] No targets computed"))
                if hasattr(self, '_create_target_buttons'):
                    self._create_target_buttons({})



        except Exception as e:
            ui_q.put(("toast", f"[Pointing] 계산 실패: {e}"))

    def pointing_move(self):
        """기존 pointing_move - 첫 번째 track으로 이동"""
        try:
            pan_t  = float(self.point_pan_target.get())
            tilt_t = float(self.point_tilt_target.get())
        except Exception:
            ui_q.put(("toast", "먼저 '가중평균 계산'으로 타깃을 구하세요."))
            return
        spd = int(100); acc = float(1.0)

        # 현재 명령 각도 기억
        self._curr_pan, self._curr_tilt = pan_t, tilt_t

        # 이동
        self.ctrl.send({"cmd":"move","pan":pan_t,"tilt":tilt_t,"speed":spd,"acc":acc})
        ui_q.put(("toast", f"→ Move to (pan={pan_t}°, tilt={tilt_t}°)"))
    
    def move_to_target(self, track_id):
        """
        특정 track_id의 계산된 pan/tilt로 카메라 이동
        
        Args:
            track_id: 이동할 track의 ID
        """
        if not hasattr(self, 'computed_targets') or track_id not in self.computed_targets:
            ui_q.put(("toast", f"❌ Track {track_id} 타깃 없음. 먼저 계산하세요."))
            return
        
        pan_t, tilt_t = self.computed_targets[track_id]
        spd = int(100); acc = float(1.0)
        
        # 현재 명령 각도 기억
        self._curr_pan, self._curr_tilt = pan_t, tilt_t
        
        # 이동
        self.ctrl.send({"cmd":"move","pan":pan_t,"tilt":tilt_t,"speed":spd,"acc":acc})
        ui_q.put(("toast", f"→ Track {track_id}: Move to (pan={pan_t}°, tilt={tilt_t}°)"))

