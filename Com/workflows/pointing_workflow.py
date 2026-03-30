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
from led_filter import (
    classify_from_single_roi,
    expand_led_roi_from_bbox,
    get_default_led_filter_params,
)


# ========== Constants ==========
CENTERING_GAIN_PAN = 0.03    # deg/px (  )
CENTERING_GAIN_TILT = 0.03   # deg/px
CONVERGENCE_TOL_PX_X = 7       #   X (px)
CONVERGENCE_TOL_PX_Y = 25      #   Y (px)
OBJECT_SIZE_CM = 5.5         #   (cm) - offset 
TARGET_OFFSET_CM = -12.25    #    12.25cm (2.75 + 5.5 + 4)
LASER_DIFF_THRESHOLD = 150   #  diff threshold ()
PHASE3_TARGET_BELOW_CM = 11.75  # 타겟 중심에서 아래(+y) 기준점(cm)
PHASE3_ENABLED = True          # Phase 3 활성화
PHASE3_MAX_ITERS = 12
PHASE3_TOL_X_PX = 6
PHASE3_TOL_Y_PX = 6
PHASE3_DIFF_TOZERO_THRESH = 70.0
PHASE3_ROI_HALF_SIZE_PX = 120
PHASE3_ROI_MARGIN_FROM_TARGET_PX = 10
PHASE3_STEP_DEG = 1.0
PHASE3_ERR_COMPARE_EPS = 0.5
PHASE3_RESPONSE_TOP_RATIO = 0.10
PHASE3_RESPONSE_COLORS = [
    (255, 0, 0),    # Blue
    (255, 255, 0),  # Cyan
    (0, 255, 0),    # Green
    (0, 255, 255),  # Yellow
    (0, 165, 255),  # Orange
    (0, 0, 255),    # Red
]
PHASE3_RESPONSE_VALUE_BINS = [0, 43, 86, 129, 172, 215, 256]
MAX_STEP_DEG = 5.0           #    (deg/step)
ROUGH_CAM_TO_LASER_CM = 6.0  #    (cm)
ROUGH_TARGET_BELOW_CM = 12.5 # YOLO    (cm)
ROUGH_PHASE1_TOL_X_PX = 15  # Rough Phase 1 X  (px)
ROUGH_PHASE1_TOL_Y_PX = 15  # Rough Phase 1 Y  (px)
ROUGH_PHASE2_START_TILT_UP_DEG = 2.0  # Phase 2 start offset (tilt up)
ROUGH_PHASE2_TILT_STEP_DEG = 1.0      # Phase 2 tilt search step (downward)
ROUGH_PHASE2_DROP_RATIO = 0.65        # /    
ROUGH_PHASE2_DROP_DELTA = 8.0         # -    
FINAL_TILT_APPROACH_UP_DEG = 1.0      #   tilt+1  
FINAL_PAN_APPROACH_RIGHT_DEG = 1.0    # scheduling test: pan+1 -> final
PHASE23_CENTER_ROI_SIZE_PX = 800      # 화면 중심 기준 Phase 2/3 유효 객체 ROI (x축 폭)


class PointingHandlerMixin:
    """Pointing mode logic - CSV analysis, regression, and laser fine-aiming"""

    @staticmethod
    def _quantize_deg(value):
        """Quantize servo command to integer degree (round)."""
        return float(int(round(float(value))))

    def _quantize_pan_tilt(self, pan, tilt):
        return self._quantize_deg(pan), self._quantize_deg(tilt)

    @staticmethod
    def _get_center_roi_box(shape, size_px=PHASE23_CENTER_ROI_SIZE_PX):
        """Return a centered x-only ROI band clipped to image bounds."""
        height, width = shape[:2]
        half = max(1, int(round(float(size_px) / 2.0)))
        cx = width // 2
        x1 = max(0, cx - half)
        x2 = min(width, cx + half)
        return (x1, 0, x2, height)

    @staticmethod
    def _get_center_roi_label(size_px=PHASE23_CENTER_ROI_SIZE_PX):
        """Return a user-facing label for the Phase 2/3 center ROI."""
        return f"CENTER ROI X {int(round(float(size_px)))}px"

    @staticmethod
    def _point_in_box(px, py, box):
        """Return True when a point lies inside the given box."""
        if box is None:
            return True
        x1, y1, x2, y2 = [float(v) for v in box]
        return x1 <= float(px) <= x2 and y1 <= float(py) <= y2

    def _bbox_center_in_box(self, bbox, box):
        """Return True when bbox center lies inside the given box."""
        if bbox is None:
            return False
        bx, by, bw, bh = [float(v) for v in bbox]
        return self._point_in_box(bx + (bw / 2.0), by + (bh / 2.0), box)

    def _get_pointing_csv_path(self):
        """Return the active pointing CSV path, if available."""
        if not hasattr(self, "point_csv_path"):
            return None
        try:
            path = self.point_csv_path.get().strip()
        except Exception:
            path = str(self.point_csv_path).strip()
        return path or None

    def _persist_final_target_to_csv(self, track_id, pan, tilt):
        """Persist the final aimed pan/tilt, LED ROI, and Phase 3 response back into the scan CSV."""
        path = self._get_pointing_csv_path()
        if not path or not os.path.exists(path):
            return False

        csv_track_ids = getattr(self, "_pointing_csv_track_ids", {}) or {}
        target_ids = tuple(int(v) for v in csv_track_ids.get(track_id, (track_id,)))
        pan_q, tilt_q = self._quantize_pan_tilt(pan, tilt)
        final_led_roi = (getattr(self, "_track_led_roi", {}) or {}).get(track_id)
        final_led_roi_source_size = (getattr(self, "_track_led_roi_source_size", {}) or {}).get(track_id)
        final_phase3_response = dict((getattr(self, "_track_phase3_response", {}) or {}).get(track_id) or {})

        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = list(reader.fieldnames or [])

            if not rows:
                return False

            if "final_pan_deg" not in fieldnames:
                fieldnames.append("final_pan_deg")
            if "final_tilt_deg" not in fieldnames:
                fieldnames.append("final_tilt_deg")
            for name in (
                "final_led_roi_x",
                "final_led_roi_y",
                "final_led_roi_w",
                "final_led_roi_h",
                "final_led_roi_src_w",
                "final_led_roi_src_h",
                "final_phase3_response_mean",
                "final_phase3_response_core",
                "final_phase3_response_max",
            ):
                if name not in fieldnames:
                    fieldnames.append(name)

            updated = 0
            for row in rows:
                try:
                    row_track_id = int(row.get("track_id", 0))
                except Exception:
                    continue
                if row_track_id not in target_ids:
                    continue
                row["final_pan_deg"] = f"{float(pan_q):.3f}"
                row["final_tilt_deg"] = f"{float(tilt_q):.3f}"
                if final_led_roi is not None and len(final_led_roi) == 4:
                    row["final_led_roi_x"] = str(int(final_led_roi[0]))
                    row["final_led_roi_y"] = str(int(final_led_roi[1]))
                    row["final_led_roi_w"] = str(int(final_led_roi[2]))
                    row["final_led_roi_h"] = str(int(final_led_roi[3]))
                else:
                    row["final_led_roi_x"] = ""
                    row["final_led_roi_y"] = ""
                    row["final_led_roi_w"] = ""
                    row["final_led_roi_h"] = ""
                if final_led_roi_source_size is not None and len(final_led_roi_source_size) == 2:
                    row["final_led_roi_src_w"] = str(int(final_led_roi_source_size[0]))
                    row["final_led_roi_src_h"] = str(int(final_led_roi_source_size[1]))
                else:
                    row["final_led_roi_src_w"] = ""
                    row["final_led_roi_src_h"] = ""
                if final_phase3_response:
                    mean_v = final_phase3_response.get("mean")
                    core_v = final_phase3_response.get("core")
                    max_v = final_phase3_response.get("max")
                    row["final_phase3_response_mean"] = "" if mean_v is None else f"{float(mean_v):.6f}"
                    row["final_phase3_response_core"] = "" if core_v is None else f"{float(core_v):.6f}"
                    row["final_phase3_response_max"] = "" if max_v is None else f"{float(max_v):.6f}"
                else:
                    row["final_phase3_response_mean"] = ""
                    row["final_phase3_response_core"] = ""
                    row["final_phase3_response_max"] = ""
                updated += 1

            if updated <= 0:
                print(f"[Pointing] CSV final target save skipped: no matching rows for UI track {track_id} -> CSV IDs {target_ids}")
                return False

            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(tmp_path, path)

            print(
                f"[Pointing] Final target saved to CSV: UI track {track_id} -> CSV IDs {target_ids}, "
                f"pan={pan_q:.1f}, tilt={tilt_q:.1f}, "
                f"roi={'set' if final_led_roi is not None else 'none'}, "
                f"phase3_score={'set' if final_phase3_response else 'none'}"
            )
            return True
        except Exception as e:
            print(f"[Pointing] Failed to save final target to CSV: {e}")
            return False
    
    # ========== CSV & Regression ==========

    def pointing_choose_csv(self):
        """CSV  """
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.point_csv_path.set(path)
            print(f"[Pointing] CSV selected: {path}")
    
    def pointing_compute(self, csv_path=None):
        """
        CSV Track ID:
          1) Tiltcx = a*pan + b pan_center = (W/2 - b)/a
          2) Pancy = e*tilt + f tilt_center = (H/2 - f)/e
         pan/tilt 
        
        Args:
            csv_path: (Optional)   CSV . GUI  .
        """
        if csv_path:
            path = csv_path
            self.point_csv_path.set(path)  # GUI  
        else:
            path = self.point_csv_path.get().strip()
            
        if not path:
            print("[Pointing] Please select a CSV file.")
            return
        
        try:
            rows = []
            W_frame = H_frame = None
            conf_min = 0.5  # Minimum confidence
            min_samples = 2  # Minimum samples for regression
            track_led_roi_samples = defaultdict(list)  # {track_id: [(x,y,w,h), ...]}
            persisted_targets = {}  # {track_id: (final_pan, final_tilt)}
            persisted_led_rois = {}  # {track_id: (x,y,w,h)}
            persisted_led_roi_source_sizes = {}  # {track_id: (W,H)}
            persisted_phase3_scores = {}  # {track_id: {"mean": float, "core": float, "max": float}}
            
            # CSV 
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for d in reader:
                    track_id_raw = d.get("track_id")
                    try:
                        track_id = int(track_id_raw)
                    except Exception:
                        track_id = None

                    W = int(d["W"]) if d.get("W") else None
                    H = int(d["H"]) if d.get("H") else None

                    if track_id is not None:
                        final_pan = d.get("final_pan_deg")
                        final_tilt = d.get("final_tilt_deg")
                        if track_id not in persisted_targets and final_pan not in ("", None) and final_tilt not in ("", None):
                            try:
                                persisted_targets[track_id] = self._quantize_pan_tilt(float(final_pan), float(final_tilt))
                            except Exception:
                                pass
                        if track_id not in persisted_led_rois:
                            try:
                                frx = int(float(d.get("final_led_roi_x", "") or 0))
                                fry = int(float(d.get("final_led_roi_y", "") or 0))
                                frw = int(float(d.get("final_led_roi_w", "") or 0))
                                frh = int(float(d.get("final_led_roi_h", "") or 0))
                                if frw > 0 and frh > 0:
                                    persisted_led_rois[track_id] = (frx, fry, frw, frh)
                                    src_w = int(float(d.get("final_led_roi_src_w", "") or (W or 0)))
                                    src_h = int(float(d.get("final_led_roi_src_h", "") or (H or 0)))
                                    if src_w > 0 and src_h > 0:
                                        persisted_led_roi_source_sizes[track_id] = (src_w, src_h)
                            except Exception:
                                pass
                        if track_id not in persisted_phase3_scores:
                            score_data = {}
                            for key, field in (
                                ("mean", "final_phase3_response_mean"),
                                ("core", "final_phase3_response_core"),
                                ("max", "final_phase3_response_max"),
                            ):
                                raw_val = d.get(field)
                                if raw_val in ("", None):
                                    continue
                                try:
                                    score_data[key] = float(raw_val)
                                except Exception:
                                    pass
                            if score_data:
                                persisted_phase3_scores[track_id] = score_data

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
                    
                    # Track ID 
                    if track_id is None:
                        continue
                    
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

                    # LED ROI ( ): schedulingIDROI  
                    try:
                        rx = int(float(d.get("led_roi_x", 0) or 0))
                        ry = int(float(d.get("led_roi_y", 0) or 0))
                        rw = int(float(d.get("led_roi_w", 0) or 0))
                        rh = int(float(d.get("led_roi_h", 0) or 0))
                        if rw > 0 and rh > 0:
                            track_led_roi_samples[track_id].append((rx, ry, rw, rh))
                    except Exception:
                        pass
            
            if not rows:
                print("[Pointing] CSV   ")
                return
            if W_frame is None or H_frame is None:
                print("[Pointing] CSVW/H  ")
                return
            
            # Track ID 
            grouped_by_track = defaultdict(list)
            for row in rows:
                grouped_by_track[row['track_id']].append(row)
            
            print(f"[Pointing] Found {len(grouped_by_track)} track(s): {list(grouped_by_track.keys())}")
            
            # track_id 
            self.computed_targets = {}  # {track_id: (pan, tilt)}
            self._pointing_gains = {}  # {track_id: (k_pan, k_tilt)}
            self._pointing_csv_track_ids = {}
            
            for track_id, track_rows in grouped_by_track.items():
                print(f"[Pointing] Computing track_id={track_id} ({len(track_rows)} detections)")
                
                # === Tilt : cx vs pan ===
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
                
                # === Pan : cy vs tilt ===
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
                
                # ===  ===
                def wavg_center(fits: dict, center_key: str):
                    if not fits:
                        return None
                    vals = np.array([fits[k][center_key] for k in fits], float)
                    w = np.array([fits[k]["N"] for k in fits], float)
                    return float(np.sum(vals * w) / np.sum(w))
                
                pan_target = wavg_center(fits_h, "pan_center")
                tilt_target = wavg_center(fits_v, "tilt_center")
                
                # Gain  (deg/px)
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
                
                regression_target = None
                if pan_target is not None and tilt_target is not None:
                    regression_target = self._quantize_pan_tilt(pan_target, tilt_target)
                    print(f"[Pointing] track_id={track_id} pan={regression_target[0]:.3f}, tilt={regression_target[1]:.3f} "
                          f"(H fits: {len(fits_h)}, V fits: {len(fits_v)}, gain: k_p={k_pan:.5f}, k_t={k_tilt:.5f})")

                persisted_target = persisted_targets.get(track_id)
                if persisted_target is not None:
                    pan_q, tilt_q = persisted_target
                    self.computed_targets[track_id] = (pan_q, tilt_q)
                    self._pointing_gains[track_id] = (k_pan, k_tilt)
                    self._pointing_csv_track_ids[track_id] = (track_id,)
                    print(
                        f"[Pointing] track_id={track_id} loaded saved final target "
                        f"pan={pan_q:.3f}, tilt={tilt_q:.3f}"
                    )
                elif regression_target is not None:
                    pan_q, tilt_q = regression_target
                    self.computed_targets[track_id] = (pan_q, tilt_q)
                    self._pointing_gains[track_id] = (k_pan, k_tilt)
                    self._pointing_csv_track_ids[track_id] = (track_id,)
                else:
                    print(f"[Pointing] track_id={track_id}   (insufficient data)")
            
            #  : 5   ID
            MERGE_TOL = 5.0  # deg
            merged = self._merge_similar_targets(
                self.computed_targets,
                grouped_by_track,
                MERGE_TOL,
                W_frame,
                H_frame,
                min_samples,
                persisted_targets=persisted_targets,
            )
            if merged:
                self.computed_targets = merged['targets']
                self._pointing_gains = merged['gains']
                self._pointing_csv_track_ids = merged.get('members', {})

            # TrackLED ROI (CSVled_roi_*  
            #   track_id , trackROI 
            track_led_roi = {}
            track_led_roi_source_size = {}
            track_phase3_response = {}
            for tid in self.computed_targets.keys():
                member_ids = tuple(int(v) for v in self._pointing_csv_track_ids.get(tid, (tid,)))
                persisted_roi = None
                persisted_size = None
                persisted_score = None
                for member_id in member_ids:
                    roi = persisted_led_rois.get(member_id)
                    if roi is not None:
                        persisted_roi = roi
                        persisted_size = persisted_led_roi_source_sizes.get(member_id)
                    score = persisted_phase3_scores.get(member_id)
                    if score is not None and persisted_score is None:
                        persisted_score = dict(score)
                    if persisted_roi is not None and persisted_score is not None:
                        break
                if persisted_roi is not None:
                    track_led_roi[int(tid)] = tuple(int(v) for v in persisted_roi)
                    if persisted_size is not None:
                        track_led_roi_source_size[int(tid)] = (
                            int(persisted_size[0]),
                            int(persisted_size[1]),
                        )
                    else:
                        track_led_roi_source_size[int(tid)] = (int(W_frame), int(H_frame))
                if persisted_score is not None:
                    track_phase3_response[int(tid)] = persisted_score

                samples = []
                for member_id in member_ids:
                    samples.extend(track_led_roi_samples.get(member_id, []))
                if persisted_roi is not None or not samples:
                    continue
                arr = np.array(samples, dtype=float)
                med = np.median(arr, axis=0)
                roi = tuple(int(v) for v in med.tolist())
                if roi[2] > 0 and roi[3] > 0:
                    track_led_roi[int(tid)] = roi
                    track_led_roi_source_size[int(tid)] = (int(W_frame), int(H_frame))
            self._track_led_roi = track_led_roi
            self._track_led_roi_source_size = track_led_roi_source_size
            self._track_phase3_response = track_phase3_response
            # Renumber final track IDs to 1..N for UI consistency
            self._renumber_computed_targets()
            # UI  ( )
            if self.computed_targets:
                print(f"[Pointing] {len(self.computed_targets)} target(s) after merge")
                if hasattr(self, '_create_target_buttons'):
                    self._create_target_buttons(self.computed_targets)
            else:
                print("[Pointing] No targets computed")
                if hasattr(self, '_create_target_buttons'):
                    self._create_target_buttons({})
        
        except Exception as e:
            print(f"[Pointing]  : {e}")
            import traceback
            traceback.print_exc()

    def _renumber_computed_targets(self):
        """Renumber computed target-related dict keys to 1..N."""
        if not hasattr(self, "computed_targets") or not self.computed_targets:
            return

        old_ids = sorted(self.computed_targets.keys())
        expected_ids = list(range(1, len(old_ids) + 1))
        if old_ids == expected_ids:
            return

        id_map = {old_id: new_id for new_id, old_id in enumerate(old_ids, start=1)}
        new_targets = {}
        new_gains = {}
        old_rois = dict(getattr(self, "_track_led_roi", {}) or {})
        old_roi_sizes = dict(getattr(self, "_track_led_roi_source_size", {}) or {})
        old_phase3_scores = dict(getattr(self, "_track_phase3_response", {}) or {})
        new_rois = {}
        new_roi_sizes = {}
        new_phase3_scores = {}
        old_csv_track_ids = dict(getattr(self, "_pointing_csv_track_ids", {}) or {})
        new_csv_track_ids = {}

        for old_id in old_ids:
            new_id = id_map[old_id]
            new_targets[new_id] = self.computed_targets[old_id]
            gain = self._pointing_gains.get(old_id, (CENTERING_GAIN_PAN, CENTERING_GAIN_TILT))
            new_gains[new_id] = gain
            if old_id in old_rois:
                new_rois[new_id] = old_rois[old_id]
            if old_id in old_roi_sizes:
                new_roi_sizes[new_id] = old_roi_sizes[old_id]
            if old_id in old_phase3_scores:
                new_phase3_scores[new_id] = dict(old_phase3_scores[old_id])
            new_csv_track_ids[new_id] = tuple(old_csv_track_ids.get(old_id, (old_id,)))

        self.computed_targets = new_targets
        self._pointing_gains = new_gains
        self._track_led_roi = new_rois
        self._track_led_roi_source_size = new_roi_sizes
        self._track_phase3_response = new_phase3_scores
        self._pointing_csv_track_ids = new_csv_track_ids
        print(f"[Pointing] ID renumbered: {id_map}")
    
    def _merge_similar_targets(self, targets, grouped_by_track, tol, W_frame, H_frame, min_samples, persisted_targets=None):
        """
        Merge nearby targets within tol and recompute representative target.
        """
        if len(targets) <= 1:
            return None

        persisted_targets = persisted_targets or {}
        
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
        
        # (  ) None 
        if all(len(g) == 1 for g in merged_groups):
            return None
        
        print(f"[Pointing]    (tol={tol}):")
        new_targets = {}
        new_gains = {}
        new_members = {}
        
        for group in merged_groups:
            rep_id = min(group)  #  ID 
            new_members[rep_id] = tuple(sorted(group))
            
            if len(group) == 1:
                #  
                new_targets[rep_id] = targets[rep_id]
                new_gains[rep_id] = self._pointing_gains.get(rep_id, (CENTERING_GAIN_PAN, CENTERING_GAIN_TILT))
                continue
            
            print(f"  IDs {group} ID {rep_id}  ")

            # If any source track in the merged group already has a saved final
            # target, keep that persisted target as the representative instead of
            # recomputing a regression target.
            group_persisted = {
                tid: persisted_targets[tid]
                for tid in sorted(group)
                if tid in persisted_targets
            }
            if group_persisted:
                saved_target = group_persisted.get(rep_id)
                if saved_target is None:
                    counts = defaultdict(int)
                    first_owner = {}
                    for tid, target in group_persisted.items():
                        counts[target] += 1
                        if target not in first_owner:
                            first_owner[target] = tid
                    saved_target = min(
                        counts.keys(),
                        key=lambda target: (-counts[target], first_owner[target]),
                    )
                new_targets[rep_id] = saved_target
                new_gains[rep_id] = self._pointing_gains.get(rep_id, (CENTERING_GAIN_PAN, CENTERING_GAIN_TILT))
                print(
                    f"  -> keep saved final target pan={saved_target[0]:.3f}, "
                    f"tilt={saved_target[1]:.3f} from CSV"
                )
                continue
            
            #   
            combined_rows = []
            for tid in group:
                if tid in grouped_by_track:
                    combined_rows.extend(grouped_by_track[tid])
            
            #   
            result = self._compute_single_target(combined_rows, W_frame, H_frame, min_samples)
            if result:
                new_targets[rep_id] = result['target']
                new_gains[rep_id] = result['gain']
                print(f"  pan={result['target'][0]:.3f}, tilt={result['target'][1]:.3f} "
                      f"({len(combined_rows)} detections)")
            else:
                #   ID 
                new_targets[rep_id] = targets[rep_id]
                new_gains[rep_id] = self._pointing_gains.get(rep_id, (CENTERING_GAIN_PAN, CENTERING_GAIN_TILT))
        
        return {'targets': new_targets, 'gains': new_gains, 'members': new_members}
    
    def _compute_single_target(self, rows, W_frame, H_frame, min_samples):
        """  pan/tilt  ()"""
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
        
        pan_q, tilt_q = self._quantize_pan_tilt(pan_t, tilt_t)
        return {'target': (pan_q, tilt_q), 'gain': (k_pan, k_tilt)}

    # ========== Laser Fine-Aiming ==========

    def set_pointing_mode(self, mode):
        """Pointing mode 설정 (adaptive 고정)."""
        mode = str(mode or "").strip().lower()
        # Backward compatibility: old "rough" or removed "legacy" are mapped.
        if mode in ("rough", "legacy"):
            mode = "adaptive"
        if mode != "adaptive":
            print(f"[Pointing] invalid mode '{mode}', fallback to adaptive")
            mode = "adaptive"
        self.pointing_mode = "adaptive"
        print("[Pointing] mode set to adaptive")
    
    def move_to_target(
        self,
        track_id,
        use_tilt_approach=False,
        use_pan_tilt_approach=False,
        pan_tilt_approach_wait_s=0.3,
    ):
        """
         track_idpan/tilt  
        """
        if not hasattr(self, 'computed_targets') or track_id not in self.computed_targets:
            print(f"[Pointing] Track {track_id} target not found. Compute targets first.")
            return
        
        #   
        if hasattr(self, '_aiming_active') and self._aiming_active:
            print(f"[Pointing]   .  .")
            return

        pan_t, tilt_t = self.computed_targets[track_id]
        pan_t, tilt_t = self._quantize_pan_tilt(pan_t, tilt_t)
        self.computed_targets[track_id] = (pan_t, tilt_t)
        self._update_target_button_value(track_id, pan_t, tilt_t)

        # Start    
        self._selected_track_id = track_id
        self._curr_pan = pan_t
        self._curr_tilt = tilt_t

        print(f"[Pointing] Track {track_id} .  : pan={pan_t}, tilt={tilt_t}")

        # Scheduling test path: pan+1/tilt+1 -> final target
        if use_pan_tilt_approach:
            self._apply_final_pan_tilt_approach(
                pan_t,
                tilt_t,
                settle_s=max(0.02, float(pan_tilt_approach_wait_s)),
            )
        #   ( tilt+1 ->  tilt )
        elif use_tilt_approach:
            self._apply_final_tilt_approach(pan_t, tilt_t, settle_s=max(0.05, self.scan_tab.settle.get()))
        else:
            spd = 100
            acc = 1.0
            self.ctrl.send({"cmd": "move", "pan": pan_t, "tilt": tilt_t, "speed": spd, "acc": acc})
        print(f"[Pointing] Track {track_id} selected and moved to initial position. Press Start Aiming.")

    def start_aiming(self, track_id=None):
        """
        track_id   
        """
        if track_id is None:
            track_id = getattr(self, '_selected_track_id', None)
        else:
            self._selected_track_id = track_id

        if track_id is None:
            print("[Pointing] Start :  ")
            return False

        if not hasattr(self, 'computed_targets') or track_id not in self.computed_targets:
            print(f"[Pointing] Start : Track {track_id} ")
            return False

        if hasattr(self, '_aiming_active') and self._aiming_active:
            print("[Pointing]   .  .")
            return False

        pan_t, tilt_t = self.computed_targets[track_id]
        pan_t, tilt_t = self._quantize_pan_tilt(pan_t, tilt_t)
        self.computed_targets[track_id] = (pan_t, tilt_t)
        self._update_target_button_value(track_id, pan_t, tilt_t)

        # Aiming  Preview  (/ 
        self._aiming_restore_preview = bool(getattr(self, 'preview_active', False))
        if self._aiming_restore_preview and hasattr(self, '_get_preview_cfg'):
            self._aiming_preview_cfg = self._get_preview_cfg()
        else:
            self._aiming_preview_cfg = None

        mode = getattr(self, "pointing_mode", "adaptive")
        if mode in ("rough", "legacy"):
            mode = "adaptive"
        if mode != "adaptive":
            mode = "adaptive"

        print(f"[Pointing] ===== Track {track_id} Fine-Aiming  (mode={mode}) =====")
        print(f"[Pointing]  : pan={pan_t}, tilt={tilt_t}")

        # IR Mode (IR Laser )
        print("[Pointing] IR Mode...")
        self.set_ir_cut("day")  # day = IR   IR  
        time.sleep(0.5)

        self._aiming_active = True
        self._aiming_cancel_event = threading.Event()
        self._aiming_track_id = track_id
        self._curr_pan = pan_t
        self._curr_tilt = tilt_t
        
        #    Event + 
        self._pointing_img_event = threading.Event()
        self._pointing_img_data = None
        
        thread_target = self._fine_aim_thread_adaptive
        t = threading.Thread(target=thread_target, args=(track_id,), daemon=True)
        t.start()
        return True

    def _restore_preview_after_aiming(self, reason="aiming"):
        """Aiming / Preview  ( ON"""
        if not getattr(self, '_aiming_restore_preview', False):
            return

        #   OFF 
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

        # Pi snap thread    
        self.root.after(400, _do_restore)
    
    def _snap_and_wait(self, label, timeout=10.0, shutter_speed=None, analogue_gain=None):
        """
        Snap     (Thread-blocking)
        """
        if not getattr(self, '_aiming_active', False):
            return None

        self._pointing_img_event.clear()
        self._pointing_img_data = None
        
        # Scan  ()
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
        
        #    
        if shutter_speed is not None:
            cmd["shutter_speed"] = int(shutter_speed)
        if analogue_gain is not None:
            cmd["analogue_gain"] = float(analogue_gain)
            
        self.ctrl.send(cmd)

        #   (stop_aiming   short-poll)
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
                print(f"[Pointing]  Snap timeout: {label}")
                return None

            if self._pointing_img_event.wait(timeout=min(poll_s, remain)):
                # stop  set() 
                if not getattr(self, '_aiming_active', False):
                    return None
                if cancel_evt is not None and cancel_evt.is_set():
                    return None
                return self._pointing_img_data
    
    def _on_pointing_image_received(self, name, data):
        """
        Pointing   (event_handlers )
        """
        try:
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(data))
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            self._pointing_img_data = bgr
            self._pointing_img_event.set()
        except Exception as e:
            print(f"[Pointing]  : {e}")
            self._pointing_img_event.set()

    def _apply_final_tilt_approach(self, pan, tilt, settle_s=0.1):
        """  tilt +1 ->  tilt  """
        pan_f = float(pan)
        tilt_f = float(tilt)
        pre_tilt = max(-30.0, min(90.0, tilt_f + FINAL_TILT_APPROACH_UP_DEG))
        wait_s = max(0.02, float(settle_s))

        # 1) 
        self.ctrl.send({
            "cmd": "move",
            "pan": pan_f,
            "tilt": pre_tilt,
            "speed": 100,
            "acc": 1.0,
        })
        time.sleep(wait_s)

    def _apply_final_pan_tilt_approach(self, pan, tilt, settle_s=0.3):
        """Scheduling test path: pan+1, tilt+1 -> final target."""
        pan_f = float(pan)
        tilt_f = float(tilt)
        pre_pan = pan_f + FINAL_PAN_APPROACH_RIGHT_DEG
        pre_tilt = max(-30.0, min(90.0, tilt_f + FINAL_TILT_APPROACH_UP_DEG))
        pre_pan, pre_tilt = self._quantize_pan_tilt(pre_pan, pre_tilt)
        wait_s = max(0.02, float(settle_s))

        self.ctrl.send({
            "cmd": "move",
            "pan": pre_pan,
            "tilt": pre_tilt,
            "speed": 100,
            "acc": 1.0,
        })
        time.sleep(wait_s)

        self.ctrl.send({
            "cmd": "move",
            "pan": pan_f,
            "tilt": tilt_f,
            "speed": 100,
            "acc": 1.0,
        })

        # 2)  tilt
        self.ctrl.send({
            "cmd": "move",
            "pan": pan_f,
            "tilt": tilt_f,
            "speed": 100,
            "acc": 1.0,
        })
        time.sleep(wait_s)

    def _fine_aim_thread_adaptive(self, track_id):
        """
        Adaptive  Thread:
          Phase 1) YOLO  X  X (X)
          Phase 2)  tilt +2deg  Laser ON/OFF(shutter=100) diff                   YOLO bbox   tilt1deg.
          Phase 3) Phase2 완료 지점에서 solarcell area 내부 laser response 최대 pan 선택.
        """
        try:
            settle = self.scan_tab.settle.get()
            led_settle = self.scan_tab.led_settle.get()

            gains = self._pointing_gains.get(track_id, (CENTERING_GAIN_PAN, CENTERING_GAIN_TILT))
            k_pan, k_tilt = gains

            tol_phase1_x = ROUGH_PHASE1_TOL_X_PX
            phase = 1
            last_px_per_cm = None
            phase1_best_pan = float(int(round(getattr(self, "_curr_pan", 0.0))))
            phase1_best_abs_err = float("inf")
            phase1_prev_sign = None
            phase1_signflip_extra_done = False
            phase1_lock_after_extra = False
            phase2_prev_mean = None
            phase2_prev_tilt = None

            print(
                f"[Pointing-Adaptive] : settle={settle}s, led_settle={led_settle}s, "
                f"k_pan={k_pan:.5f}, k_tilt={k_tilt:.5f}, "
                f"tol_p1_x={tol_phase1_x}px, phase2_start_up={ROUGH_PHASE2_START_TILT_UP_DEG}deg, "
                f"phase2_step={ROUGH_PHASE2_TILT_STEP_DEG}deg, drop_ratio={ROUGH_PHASE2_DROP_RATIO}, "
                f"drop_delta={ROUGH_PHASE2_DROP_DELTA}"
            )

            time.sleep(max(settle, 1.0))
            self.ctrl.send({"cmd": "laser", "value": 0})
            self._update_aiming_status(track_id, 0, "Adaptive : YOLO   ")

            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"Captures/Pointing/{now_str}_Track_{track_id}_adaptive"
            os.makedirs(log_dir, exist_ok=True)
            print(f"[Pointing-Adaptive] Logging to: {log_dir}")

            iteration = 0
            while self._aiming_active:
                iteration += 1
                phase_name = "CenterX" if phase == 1 else "Brightness"
                print(f"\n[Pointing-Adaptive] ===== Iteration {iteration} / Phase {phase_name} =====")
                self._update_aiming_status(track_id, iteration, f"Adaptive  {iteration} ({phase_name})")

                # Step 1: YOLO   LED diff)
                # LED  Normal(  
                self.set_ir_cut("night")
                time.sleep(0.05)
                self.ctrl.send({"cmd": "led", "value": 255})
                time.sleep(led_settle)
                img_led_on = self._snap_and_wait(
                    f"pointing_adaptive_led_on_{iteration}",
                    shutter_speed=10000,
                    analogue_gain=None,
                )
                if img_led_on is None:
                    self.ctrl.send({"cmd": "led", "value": 0})
                    self.set_ir_cut("day")
                    print("[Pointing-Adaptive]  LED ON   ")
                    continue

                self.ctrl.send({"cmd": "led", "value": 0})
                time.sleep(led_settle)
                img_led_off = self._snap_and_wait(
                    f"pointing_adaptive_led_off_{iteration}",
                    shutter_speed=10000,
                    analogue_gain=None,
                )
                if img_led_off is None:
                    self.set_ir_cut("day")
                    print("[Pointing-Adaptive]  LED OFF   ")
                    continue
                
                #   IR  
                self.set_ir_cut("day")

                try:
                    cv2.imwrite(f"{log_dir}/iter_{iteration}_led_on.jpg", img_led_on)
                    cv2.imwrite(f"{log_dir}/iter_{iteration}_led_off.jpg", img_led_off)
                except Exception as e:
                    print(f"[Pointing-Adaptive] Log save failed: {e}")

                phase_center_roi_box = self._get_center_roi_box(img_led_on.shape) if phase >= 2 else None
                obj_cx, obj_cy, bbox, all_bboxes = self._find_object_center(
                    img_led_on,
                    img_led_off,
                    selection_roi_box=phase_center_roi_box,
                )
                if obj_cx is None:
                    if phase >= 2:
                        roi_msg = (
                            f"[Iter {iteration}] Phase 2 object must stay inside "
                            f"center ROI x {PHASE23_CENTER_ROI_SIZE_PX}px"
                        )
                        detect_debug = img_led_on.copy()
                        rx1, ry1, rx2, ry2 = [int(v) for v in phase_center_roi_box]
                        cv2.rectangle(detect_debug, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
                        cv2.putText(detect_debug, self._get_center_roi_label(), (rx1, max(20, ry1 + 24)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                        if all_bboxes:
                            for mx, my, mw, mh in all_bboxes:
                                cv2.rectangle(
                                    detect_debug,
                                    (int(mx), int(my)),
                                    (int(mx + mw), int(my + mh)),
                                    (100, 100, 100),
                                    1,
                                )
                        self.root.after(
                            0,
                            lambda img=detect_debug.copy(), msg=roi_msg:
                            self._show_debug_preview(img, iteration=iteration, status_text=msg, status_color="orange"),
                        )
                        self._update_aiming_status(
                            track_id,
                            iteration,
                            f"Phase 2: center ROI x {PHASE23_CENTER_ROI_SIZE_PX}px 밖 객체는 제외",
                        )
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
                        time.sleep(settle)
                        continue
                    print("[Pointing-Adaptive]   ")
                    continue
                H, W = img_led_on.shape[:2]
                led_roi = getattr(self, "_last_object_led_info", {}).get("roi")
                if led_roi is not None and hasattr(self, "_track_led_roi"):
                    try:
                        self._track_led_roi[track_id] = tuple(int(v) for v in led_roi)
                        if hasattr(self, "_track_led_roi_source_size"):
                            self._track_led_roi_source_size[track_id] = (int(W), int(H))
                    except Exception:
                        pass
                frame_cx = W / 2.0
                frame_cy = H / 2.0

                if bbox and bbox[2] > 0:
                    px_per_cm = float(bbox[2]) / OBJECT_SIZE_CM
                    last_px_per_cm = px_per_cm
                elif last_px_per_cm is not None:
                    px_per_cm = last_px_per_cm
                else:
                    # bbox  fallback
                    px_per_cm = 10.0

                if phase == 1:
                    # Phase 1: YOLO  X  X (X only)
                    target_x = frame_cx
                    target_y = frame_cy
                    ref_x = obj_cx
                    ref_y = obj_cy
                    err_x = obj_cx - frame_cx

                    print(f"[Pointing-Adaptive] CenterX: err_x={err_x:.1f}px, px_per_cm={px_per_cm:.3f}")

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
                        f"Adaptive CenterX: err_x={err_x:.1f}px (tol_x={tol_phase1_x}px)",
                    )

                    cur_abs_err = abs(err_x)
                    if cur_abs_err < phase1_best_abs_err:
                        phase1_best_abs_err = cur_abs_err
                        phase1_best_pan = float(int(round(self._curr_pan)))

                    ctrl_sign = 1 if (err_x * k_pan) > 0 else (-1 if (err_x * k_pan) < 0 else 0)
                    phase1_crossed = (
                        phase1_prev_sign is not None
                        and ctrl_sign != 0
                        and phase1_prev_sign != 0
                        and ctrl_sign != phase1_prev_sign
                    )

                    # 요청사항: 부호가 바뀌면 한 번만 기존 방향으로 1스텝 더 진행 후 재평가
                    if (
                        phase1_crossed
                        and not phase1_signflip_extra_done
                        and phase1_prev_sign in (-1, 1)
                    ):
                        extra_dir = int(phase1_prev_sign)
                        cur_pan_int = int(self._curr_pan)
                        extra_pan = float(max(-180, min(180, cur_pan_int + extra_dir)))
                        if int(extra_pan) != cur_pan_int:
                            self._curr_pan = extra_pan
                            self.ctrl.send(
                                {
                                    "cmd": "move",
                                    "pan": self._curr_pan,
                                    "tilt": self._curr_tilt,
                                    "speed": 100,
                                    "acc": 1.0,
                                }
                            )
                            phase1_signflip_extra_done = True
                            phase1_lock_after_extra = True
                            # 교차를 소비 처리: 다음 iteration에서 새 상태로 재평가
                            phase1_prev_sign = ctrl_sign
                            print(
                                f"[Pointing-Adaptive] Phase 1 sign-cross -> extra step "
                                f"prev_dir={extra_dir:+d}, pan={int(self._curr_pan)}"
                            )
                            self._update_aiming_status(
                                track_id,
                                iteration,
                                f"Phase1 sign-cross extra step: pan={int(self._curr_pan)}",
                            )
                            time.sleep(settle)
                            continue

                    if phase1_crossed or phase1_lock_after_extra:
                        lock_pan = float(int(round(phase1_best_pan)))
                        self._curr_pan = lock_pan
                        lock_reason = "cross+extra" if phase1_lock_after_extra and not phase1_crossed else "cross"
                        phase1_lock_after_extra = False
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
                            f"[Pointing-Adaptive] Phase 1 lock ({lock_reason}), "
                            f"lock pan={lock_pan:.0f}, best |err_x|={phase1_best_abs_err:.1f}px"
                        )
                        self._update_aiming_status(
                            track_id,
                            iteration,
                            f"Phase1 lock ({lock_reason}) -> pan={lock_pan:.0f}, go Phase2",
                        )
                        time.sleep(settle)

                        phase = 2
                        # Phase 2 starts immediately after Phase 1 finish (no LR probe).
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
                            "[Pointing-Adaptive] Phase 1(X) converged -> Phase 2 start "
                            f"(tilt +{ROUGH_PHASE2_START_TILT_UP_DEG}deg)"
                        )
                        self._update_aiming_status(track_id, iteration, "Phase 1(X) done, then Phase 2")
                        time.sleep(settle)
                        continue

                    # Phase 1 step search (1 degree): follow sign of d_pan = err_x * k_pan
                    step_dir = ctrl_sign
                    cur_pan_int = int(self._curr_pan)
                    next_pan = float(max(-180, min(180, cur_pan_int + step_dir)))
                    next_tilt = self._curr_tilt
                    phase1_prev_sign = ctrl_sign

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
                                "step={sd:+d} Next=({np:.3f},{nt:.3f})\n".format(
                                    it=iteration,
                                    ox=obj_cx,
                                    ex=err_x,
                                    sd=step_dir,
                                    np=next_pan,
                                    nt=next_tilt,
                                )
                            )
                    except Exception as e:
                        print(f"[Pointing-Adaptive] Log write failed: {e}")

                    time.sleep(settle)
                    continue

                # Phase 2: Laser ON/OFF diff YOLO bbox    
                print("[Pointing-Adaptive] Phase 2: Laser ON...")
                self.ctrl.send({"cmd": "laser", "value": 1})
                time.sleep(led_settle)
                img_laser_on = self._snap_and_wait(
                    f"pointing_adaptive_laser_on_{iteration}",
                    shutter_speed=100,
                    analogue_gain=1.0,
                )
                if img_laser_on is None:
                    print("[Pointing-Adaptive]  Laser ON   ")
                    continue

                print("[Pointing-Adaptive] Phase 2: Laser OFF...")
                self.ctrl.send({"cmd": "laser", "value": 0})
                time.sleep(led_settle)
                img_laser_off = self._snap_and_wait(
                    f"pointing_adaptive_laser_off_{iteration}",
                    shutter_speed=100,
                    analogue_gain=1.0,
                )
                if img_laser_off is None:
                    print("[Pointing-Adaptive]  Laser OFF   ")
                    continue

                try:
                    cv2.imwrite(f"{log_dir}/iter_{iteration}_laser_on.jpg", img_laser_on)
                    cv2.imwrite(f"{log_dir}/iter_{iteration}_laser_off.jpg", img_laser_off)
                except Exception as e:
                    print(f"[Pointing-Adaptive] Log save failed: {e}")

                laser_diff = cv2.absdiff(img_laser_on, img_laser_off)
                laser_gray = cv2.cvtColor(laser_diff, cv2.COLOR_BGR2GRAY)

                if not bbox:
                    print("[Pointing-Adaptive]  Phase 2: YOLO bbox ,   ")
                    time.sleep(settle)
                    continue

                bx, by, bw, bh = [int(v) for v in bbox]
                x1 = max(0, bx)
                y1 = max(0, by)
                x2 = min(W, bx + bw)
                y2 = min(H, by + bh)
                if x2 <= x1 or y2 <= y1:
                    print("[Pointing-Adaptive]  Phase 2: YOLO bbox ROI  ")
                    time.sleep(settle)
                    continue

                roi = laser_gray[y1:y2, x1:x2]
                mean_bright = float(np.mean(roi)) if roi.size > 0 else 0.0

                print(
                    f"[Pointing-Adaptive] Brightness(BBox ROI {x1}:{x2}, {y1}:{y2}) "
                    f"mean={mean_bright:.2f}"
                )

                self._draw_phase2_debug_image(
                    img_led_on=img_led_on,
                    img_laser_on=img_laser_on,
                    img_laser_off=img_laser_off,
                    bbox=bbox,
                    all_bboxes=all_bboxes,
                    center_roi_box=phase_center_roi_box,
                    iteration=iteration,
                    mean_bright=mean_bright,
                    roi_ok=True,
                    log_dir=log_dir,
                )

                self._update_aiming_status(
                    track_id,
                    iteration,
                    f"Adaptive Brightness: mean={mean_bright:.1f} (bbox)",
                )

                #  tilt    tilt(   
                if phase2_prev_mean is not None:
                    drop_delta = phase2_prev_mean - mean_bright
                    drop_ratio = mean_bright / max(phase2_prev_mean, 1e-6)
                    is_drop = (drop_ratio <= ROUGH_PHASE2_DROP_RATIO) and (drop_delta >= ROUGH_PHASE2_DROP_DELTA)

                    if is_drop:
                        final_pan = self._curr_pan
                        final_tilt = self._curr_tilt
                        if not PHASE3_ENABLED:
                            final_pan = float(max(-180.0, min(180.0, float(final_pan) - 1.0)))
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
                        pan_q, tilt_q = self._quantize_pan_tilt(final_pan, final_tilt)
                        self.computed_targets[track_id] = (pan_q, tilt_q)
                        self._update_target_button_value(track_id, pan_q, tilt_q)
                        if PHASE3_ENABLED:
                            self.ctrl.send({"cmd": "laser", "value": 1})
                            phase3_target_x = float(obj_cx)
                            phase3_target_y = float(obj_cy + (PHASE3_TARGET_BELOW_CM * px_per_cm))
                            print(
                                "[Pointing-Adaptive] Phase 3 start: "
                                f"target=({phase3_target_x:.1f},{phase3_target_y:.1f}), "
                                f"px_per_cm={px_per_cm:.3f}"
                            )
                            phase3_ok, phase3_best = self._phase3_refine_laser_target(
                                track_id=track_id,
                                target_x=phase3_target_x,
                                target_y=phase3_target_y,
                                all_bboxes=all_bboxes,
                                k_pan=k_pan,
                                k_tilt=k_tilt,
                                settle=settle,
                                led_settle=led_settle,
                                log_dir=log_dir,
                                base_iteration=iteration,
                                initial_px_per_cm=px_per_cm,
                            )
                            if phase3_best is not None:
                                best_pan, best_tilt, best_resp_mean, best_resp_core, best_resp_max = phase3_best
                                self._curr_pan = float(best_pan)
                                self._curr_tilt = float(best_tilt)
                                self.ctrl.send(
                                    {
                                        "cmd": "move",
                                        "pan": self._curr_pan,
                                        "tilt": self._curr_tilt,
                                        "speed": 100,
                                        "acc": 1.0,
                                    }
                                )
                                pan_q, tilt_q = self._quantize_pan_tilt(self._curr_pan, self._curr_tilt)
                                self.computed_targets[track_id] = (pan_q, tilt_q)
                                self._update_target_button_value(track_id, pan_q, tilt_q)
                                if hasattr(self, "_track_phase3_response"):
                                    self._track_phase3_response[track_id] = {
                                        "mean": float(best_resp_mean),
                                        "core": float(best_resp_core),
                                        "max": float(best_resp_max),
                                    }
                                self._persist_final_target_to_csv(track_id, pan_q, tilt_q)
                                print(
                                    "[Pointing-Adaptive] Phase 3 best: "
                                    f"mean={best_resp_mean:.1f}, core={best_resp_core:.1f}, "
                                    f"max={best_resp_max:.1f}, "
                                    f"pose=({self._curr_pan:.2f},{self._curr_tilt:.2f}), ok={phase3_ok}"
                                )
                            else:
                                if hasattr(self, "_track_phase3_response"):
                                    self._track_phase3_response.pop(track_id, None)
                                self._persist_final_target_to_csv(track_id, final_pan, final_tilt)
                        else:
                            if hasattr(self, "_track_phase3_response"):
                                self._track_phase3_response.pop(track_id, None)
                            self._persist_final_target_to_csv(track_id, final_pan, final_tilt)
                            print(
                                "[Pointing-Adaptive] Phase 3 disabled -> "
                                f"use Phase 2 final pose=({final_pan:.2f},{final_tilt:.2f})"
                            )
                        print(
                            f"[Pointing-Adaptive] Phase 2 :    "
                            f"(prev={phase2_prev_mean:.2f}, cur={mean_bright:.2f}, "
                            f"ratio={drop_ratio:.3f}, delta={drop_delta:.2f})"
                        )
                        self._update_aiming_status(track_id, iteration, "Adaptive :   ")
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
                            print(f"[Pointing-Adaptive] Log write failed: {e}")
                        break

                #   tilt1  
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
                    print(f"[Pointing-Adaptive] Log write failed: {e}")

                time.sleep(settle)

        except Exception as e:
            print(f"[Pointing-Adaptive] : {e}")
            import traceback
            traceback.print_exc()
            self._update_aiming_status(track_id, 0, f": {e}")

        finally:
            self._aiming_active = False
            self._aiming_track_id = None
            self._restore_preview_after_aiming(reason="aiming-end")
            print("[Pointing-Adaptive] Thread ")

    def _update_aiming_status(self, track_id, iteration, message):
        """UI   (thread-safe)"""
        try:
            self.root.after(0, lambda: self.info_label.config(
                text=f" Track {track_id} [{iteration}]: {message}"
            ))
            if hasattr(self, 'pointing_tab') and hasattr(self.pointing_tab, 'update_aim_status'):
                self.root.after(0, lambda: self.pointing_tab.update_aim_status(
                    track_id, iteration, message
                ))
        except Exception:
            pass

    def _update_target_button_value(self, track_id, pan, tilt):
        """Pointing ID    (thread-safe)"""
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
          Pointing   + 
        """
        try:
            debug = base_img.copy()
            H, W = debug.shape[:2]
            err_mag = (err_x**2 + err_y**2)**0.5
            
            #  (
            tx, ty = int(target_cx), int(target_cy)
            cv2.circle(debug, (tx, ty), 12, (0, 0, 255), 3)
            cv2.drawMarker(debug, (tx, ty), (0, 0, 255), cv2.MARKER_CROSS, 50, 3)
            cv2.putText(debug, "TARGET", (tx+15, ty-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            #   (
            lx, ly = int(laser_pos[0]), int(laser_pos[1])
            cv2.circle(debug, (lx, ly), 12, (0, 255, 0), 3)
            cv2.drawMarker(debug, (lx, ly), (0, 255, 0), cv2.MARKER_CROSS, 50, 3)
            cv2.putText(debug, "LASER", (lx+15, ly-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            #  (  )
            cv2.line(debug, (tx, ty), (lx, ly), (255, 255, 255), 1, cv2.LINE_AA)
            
            # Best  BBox (
            if best_bbox:
                bx, by, bw, bh = [int(v) for v in best_bbox]
                cv2.rectangle(debug, (bx, by), (bx+bw, by+bh), (0, 255, 255), 3)
                cv2.putText(debug, "OBJECT", (bx, by-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            #  bbox()
            if all_bboxes:
                for (mx, my, mw, mh) in all_bboxes:
                    cv2.rectangle(debug, (int(mx), int(my)), 
                                  (int(mx+mw), int(my+mh)), (128, 128, 128), 2)
            
            if tol_x is None:
                tol_x = CONVERGENCE_TOL_PX_X
            if tol_y is None:
                tol_y = CONVERGENCE_TOL_PX_Y

            #  
            converged = abs(err_x) <= tol_x and abs(err_y) <= tol_y
            color = (0, 255, 0) if converged else (0, 0, 255)
            cv2.putText(debug, f"Iter {iteration}  Err: ({err_x:.1f}, {err_y:.1f}) = {err_mag:.1f}px",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(debug, f"Tol: ({tol_x}, {tol_y})px",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            #  Pan/Tilt 
            cur_pan = getattr(self, '_curr_pan', 0.0)
            cur_tilt = getattr(self, '_curr_tilt', 0.0)
            cv2.putText(debug, f"Pan: {cur_pan:.2f}  Tilt: {cur_tilt:.2f}",
                        (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)
            
            # 400x400 Crop ()
            crop_half = 200
            y1c = max(0, ty - crop_half)
            y2c = min(H, ty + crop_half)
            x1c = max(0, tx - crop_half)
            x2c = min(W, tx + crop_half)
            crop = debug[y1c:y2c, x1c:x2c]
            
            # UI (thread-safe)
            self.root.after(
                0,
                lambda img=crop.copy(), ex=err_x, ey=err_y, em=err_mag, it=iteration, tx=tol_x, ty=tol_y:
                self._show_debug_preview(img, ex, ey, em, it, tx, ty),
            )
            
            # Laser Diff   + UI 
            if img_laser_on is not None and img_laser_off is not None:
                # 1. Diff
                laser_diff = cv2.absdiff(img_laser_on, img_laser_off)
                laser_gray = cv2.cvtColor(laser_diff, cv2.COLOR_BGR2GRAY)
                
                # 2. Threshold (  50)
                _, laser_mask = cv2.threshold(laser_gray, 50, 255, cv2.THRESH_BINARY)
                
                # 3. Masking (  )
                if all_bboxes:
                    for (mx, my, mw, mh) in all_bboxes:
                        x1, y1 = int(mx), int(my)
                        x2, y2 = int(mx+mw), int(my+mh)
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(W, x2); y2 = min(H, y2)
                        
                        if x1 < x2 and y1 < y2:
                            laser_mask[y1:y2, x1:x2] = 0
                
                # 4. (Binary Mask   
                laser_vis = cv2.cvtColor(laser_mask, cv2.COLOR_GRAY2BGR)
                
                #    ()
                lx, ly = int(laser_pos[0]), int(laser_pos[1])
                cv2.circle(laser_vis, (lx, ly), 10, (0, 255, 0), 2)
                cv2.drawMarker(laser_vis, (lx, ly), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
                cv2.putText(laser_vis, f"LASER ({lx},{ly})", (lx+15, ly-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                #  ()
                cv2.circle(laser_vis, (tx, ty), 8, (0, 0, 255), 2)
                
                #   ( )
                if all_bboxes:
                    for (mx, my, mw, mh) in all_bboxes:
                        cv2.rectangle(laser_vis, (int(mx), int(my)),
                                      (int(mx+mw), int(my+mh)), (0, 0, 128), 1)
                        cv2.putText(laser_vis, "MASKED", (int(mx), int(my)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 128), 1)
                                    
                # Detection ROI  (Cyan) -  
                roi_size = 300
                cx_img, cy_img = W // 2, H // 2
                rx1 = max(0, cx_img - roi_size)
                rx2 = min(W, cx_img + roi_size)
                ry1 = max(0, cy_img - roi_size - 100)
                ry2 = min(H, cy_img + roi_size)
                
                cv2.rectangle(laser_vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
                cv2.putText(laser_vis, "ROI AREA", (rx1, ry1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                    
                # Wide Crop (1200x900) -> 400x300  (Zoom Out )
                H_l, W_l = laser_vis.shape[:2]
                y1l = max(0, ty - 450)
                y2l = min(H_l, ty + 450)
                x1l = max(0, tx - 600)
                x2l = min(W_l, tx + 600)
                laser_crop = laser_vis[y1l:y2l, x1l:x2l]
                self.root.after(0, lambda img=laser_crop.copy(): self._show_laser_diff(img))
            
            # [LOG]  
            if log_dir:
                try:
                    cv2.imwrite(f"{log_dir}/iter_{iteration}_debug.jpg", debug)
                except Exception as e:
                    print(f"[Pointing] Debug image save failed: {e}")
            
        except Exception as e:
            print(f"[Pointing] Debug   : {e}")
    
    def _show_debug_preview(self, img_bgr, err_x=0, err_y=0, err_mag=0, iteration=0,
                            tol_x=None, tol_y=None, status_text=None, status_color=None):
        """Pointing   (main thread)"""
        try:
            from PIL import Image, ImageTk
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb).resize((400, 400), Image.LANCZOS)
            photo = ImageTk.PhotoImage(im)
            
            if hasattr(self, 'pointing_tab') and hasattr(self.pointing_tab, 'debug_preview_label'):
                self.pointing_tab.debug_preview_label.config(image=photo)
                self.pointing_tab.debug_preview_label.image = photo
            
            #  
            if hasattr(self, 'pointing_tab') and hasattr(self.pointing_tab, 'debug_error_label'):
                if status_text is not None:
                    text = str(status_text)
                    color = status_color or "#888"
                else:
                    if tol_x is None:
                        tol_x = CONVERGENCE_TOL_PX_X
                    if tol_y is None:
                        tol_y = CONVERGENCE_TOL_PX_Y
                    color = "green" if abs(err_x) <= tol_x and abs(err_y) <= tol_y else "red"
                    text = (
                        f"[Iter {iteration}]  err_x={err_x:.1f}px  err_y={err_y:.1f}px  "
                        f"|err|={err_mag:.1f}px  (tol=({tol_x}, {tol_y})px)"
                    )
                self.pointing_tab.debug_error_label.config(
                    text=text,
                    fg=color
                )
        except Exception as e:
            print(f"[Pointing] Debug preview : {e}")
    
    def _show_laser_diff(self, img_bgr):
        """Pointing 탭 하단 보조 패널 이미지 표시 (Phase 2/3 debug 공용)."""
        try:
            from PIL import Image, ImageTk
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb).resize((400, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(im)
            
            if hasattr(self, 'pointing_tab') and hasattr(self.pointing_tab, 'laser_diff_label'):
                self.pointing_tab.laser_diff_label.config(image=photo)
                self.pointing_tab.laser_diff_label.image = photo
        except Exception as e:
            print(f"[Pointing] Laser diff preview : {e}")

    def _draw_phase2_debug_image(
        self,
        img_led_on,
        img_laser_on,
        img_laser_off,
        bbox,
        all_bboxes,
        center_roi_box,
        iteration,
        mean_bright,
        roi_ok,
        log_dir=None,
    ):
        """Render Phase 2 detection/laser diff overlays into the Pointing GUI."""
        try:
            detect_debug = img_led_on.copy()
            diff = cv2.absdiff(img_laser_on, img_laser_off)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            laser_debug = cv2.cvtColor(diff_gray, cv2.COLOR_GRAY2BGR)

            cx_img = detect_debug.shape[1] // 2
            cy_img = detect_debug.shape[0] // 2
            rx1, ry1, rx2, ry2 = [int(v) for v in center_roi_box]

            for canvas in (detect_debug, laser_debug):
                cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
                cv2.putText(
                    canvas,
                    self._get_center_roi_label(),
                    (rx1, max(20, ry1 + 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
                cv2.drawMarker(canvas, (cx_img, cy_img), (255, 255, 255), cv2.MARKER_CROSS, 28, 2)

            if all_bboxes:
                for mx, my, mw, mh in all_bboxes:
                    p1 = (int(mx), int(my))
                    p2 = (int(mx + mw), int(my + mh))
                    cv2.rectangle(detect_debug, p1, p2, (100, 100, 100), 1)
                    cv2.rectangle(laser_debug, p1, p2, (100, 100, 100), 1)

            if bbox:
                bx, by, bw, bh = [int(v) for v in bbox]
                obj_center = (int(round(bx + (bw / 2.0))), int(round(by + (bh / 2.0))))
                roi_color = (0, 255, 0) if roi_ok else (0, 140, 255)
                cv2.rectangle(detect_debug, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
                cv2.rectangle(laser_debug, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
                cv2.drawMarker(detect_debug, obj_center, roi_color, cv2.MARKER_TILTED_CROSS, 24, 2)
                cv2.drawMarker(laser_debug, obj_center, roi_color, cv2.MARKER_TILTED_CROSS, 24, 2)
                cv2.putText(detect_debug, "OBJECT ROI", (bx, max(20, by - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(laser_debug, "BBOX ROI", (bx, max(20, by - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            cv2.putText(detect_debug, f"Phase 2 Detection | Iter {iteration}", (20, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)
            cv2.putText(
                detect_debug,
                f"ROI check: {'OK' if roi_ok else 'OUT'}",
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if roi_ok else (0, 140, 255),
                2,
            )
            cv2.putText(detect_debug, f"Pan={float(getattr(self, '_curr_pan', 0.0)):.2f} Tilt={float(getattr(self, '_curr_tilt', 0.0)):.2f}",
                        (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)

            cv2.putText(laser_debug, f"Phase 2 Laser Diff | Iter {iteration}", (20, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)
            cv2.putText(laser_debug, f"Brightness mean={float(mean_bright):.1f}", (20, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(
                laser_debug,
                f"ROI check: {'OK' if roi_ok else 'OUT'}",
                (20, 102),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if roi_ok else (0, 140, 255),
                2,
            )

            status_text = (
                f"[Iter {iteration}] Phase 2 mean={float(mean_bright):.1f} | "
                f"center ROI X {'OK' if roi_ok else 'OUT'}"
            )
            self.root.after(
                0,
                lambda img=detect_debug.copy(), msg=status_text, ok=roi_ok:
                self._show_debug_preview(img, iteration=iteration, status_text=msg, status_color=("green" if ok else "orange")),
            )
            self.root.after(0, lambda img=laser_debug.copy(): self._show_laser_diff(img))

            if log_dir:
                cv2.imwrite(f"{log_dir}/iter_{iteration}_phase2_detect_debug.jpg", detect_debug)
                cv2.imwrite(f"{log_dir}/iter_{iteration}_phase2_laser_debug.jpg", laser_debug)
        except Exception as e:
            print(f"[Pointing] Phase2 debug render failed: {e}")

    @staticmethod
    def _phase3_clip_box(area_box, shape):
        """Clip a box to image bounds."""
        height, width = shape[:2]
        x1, y1, x2, y2 = [int(v) for v in area_box]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        return x1, y1, x2, y2

    def _phase3_build_area_geometry(self, bbox, fallback_target, px_per_cm_hint):
        """Build the solarcell-area target geometry used for Phase 3 response scoring."""
        if bbox and bbox[2] > 0 and bbox[3] > 0:
            bx, by, bw, bh = [float(v) for v in bbox]
            obj_cx = bx + (bw / 2.0)
            obj_cy = by + (bh / 2.0)
            px_per_cm_x = bw / OBJECT_SIZE_CM
            px_per_cm_y = bh / OBJECT_SIZE_CM
            px_per_cm = max(1.0, (px_per_cm_x + px_per_cm_y) / 2.0)
            target_x = float(obj_cx)
            target_y = float(obj_cy + (PHASE3_TARGET_BELOW_CM * px_per_cm))
        else:
            px_per_cm = float(px_per_cm_hint) if px_per_cm_hint else 10.0
            target_x = float(fallback_target[0])
            target_y = float(fallback_target[1])

        area_side_px = float(8.0 * px_per_cm)
        half_side_px = area_side_px / 2.0
        area_box = (
            int(round(target_x - half_side_px)),
            int(round(target_y - half_side_px)),
            int(round(target_x + half_side_px)),
            int(round(target_y + half_side_px)),
        )
        return {
            "target_x": float(target_x),
            "target_y": float(target_y),
            "px_per_cm": float(px_per_cm),
            "area_box": area_box,
        }

    def _phase3_compute_response_metrics(self, img_laser_on, img_laser_off, area_box):
        """Compute Phase 3 response inside the 8cm x 8cm target area."""
        gray_on = cv2.cvtColor(img_laser_on, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_off = cv2.cvtColor(img_laser_off, cv2.COLOR_BGR2GRAY).astype(np.float32)
        delta_pos = np.clip(gray_on - gray_off, 0.0, None)
        delta_u8 = np.clip(delta_pos, 0.0, 255.0).astype(np.uint8)

        x1, y1, x2, y2 = self._phase3_clip_box(area_box, delta_u8.shape)
        if x2 <= x1 or y2 <= y1:
            return delta_u8, {
                "mean_delta": 0.0,
                "core_delta": 0.0,
                "max_delta": 0.0,
                "peak_pos": None,
                "clipped_area_box": (x1, y1, x2, y2),
            }

        roi = delta_pos[y1:y2, x1:x2]
        positive = roi[roi > 0]

        if positive.size <= 0:
            metrics = {
                "mean_delta": 0.0,
                "core_delta": 0.0,
                "max_delta": 0.0,
                "peak_pos": None,
                "clipped_area_box": (x1, y1, x2, y2),
            }
        else:
            top_count = max(1, int(np.ceil(float(positive.size) * PHASE3_RESPONSE_TOP_RATIO)))
            top_values = np.partition(positive, -top_count)[-top_count:]
            peak_local = np.unravel_index(np.argmax(roi), roi.shape)
            peak_pos = (int(x1 + peak_local[1]), int(y1 + peak_local[0]))
            metrics = {
                "mean_delta": float(np.mean(positive)),
                "core_delta": float(np.mean(top_values)),
                "max_delta": float(np.max(positive)),
                "peak_pos": peak_pos,
                "clipped_area_box": (x1, y1, x2, y2),
            }

        return delta_u8, metrics

    def _phase3_render_response_overlay(
        self,
        led_diff_bgr,
        area_box,
        response_u8,
        bbox,
        all_bboxes,
        center_roi_box,
        target_x,
        target_y,
        phase3_iter,
        phase3_tag,
        response_metrics,
    ):
        """Render the LED diff panel with absolute 0..255 laser-response colors."""
        base = led_diff_bgr.copy()
        overlay = base.copy()
        x1, y1, x2, y2 = response_metrics.get("clipped_area_box", self._phase3_clip_box(area_box, response_u8.shape))

        if x2 > x1 and y2 > y1:
            color_layer = np.zeros_like(base)
            for idx in range(len(PHASE3_RESPONSE_VALUE_BINS) - 1):
                low = PHASE3_RESPONSE_VALUE_BINS[idx]
                high = PHASE3_RESPONSE_VALUE_BINS[idx + 1]
                sel = np.zeros(response_u8.shape, dtype=bool)
                roi = response_u8[y1:y2, x1:x2]
                sel_roi = (roi >= low) & (roi < high)
                sel[y1:y2, x1:x2] = sel_roi
                color_layer[sel] = PHASE3_RESPONSE_COLORS[idx]

            blended = cv2.addWeighted(base, 0.55, color_layer, 0.45, 0)
            overlay[y1:y2, x1:x2] = blended[y1:y2, x1:x2]

        if all_bboxes:
            for mx, my, mw, mh in all_bboxes:
                cv2.rectangle(
                    overlay,
                    (int(mx), int(my)),
                    (int(mx + mw), int(my + mh)),
                    (100, 100, 100),
                    1,
                )
        if bbox:
            bx, by, bw, bh = [int(v) for v in bbox]
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
            cv2.putText(overlay, "OBJECT", (bx, max(20, by - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        if center_roi_box:
            rx1, ry1, rx2, ry2 = [int(v) for v in center_roi_box]
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
            cv2.putText(overlay, self._get_center_roi_label(), (rx1, max(20, ry1 + 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        tx = int(round(float(target_x)))
        ty = int(round(float(target_y)))
        cv2.drawMarker(overlay, (tx, ty), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 28, 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)

        peak = response_metrics.get("peak_pos")
        if peak is not None:
            cv2.circle(overlay, peak, 10, (0, 0, 255), 2)
            cv2.drawMarker(overlay, peak, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)

        cv2.putText(overlay, f"Phase3 {phase3_iter}-{phase3_tag}", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)
        cv2.putText(overlay, "LED DIFF + response overlay", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(
            overlay,
            "Resp mean={:.1f} core={:.1f} max={:.1f}".format(
                float(response_metrics.get("mean_delta", 0.0)),
                float(response_metrics.get("core_delta", 0.0)),
                float(response_metrics.get("max_delta", 0.0)),
            ),
            (20, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return overlay

    def _save_phase3_debug_artifacts(
        self,
        log_dir,
        base_iteration,
        phase3_iter,
        phase3_tag,
        img_led_on,
        img_led_off,
        img_laser_on,
        img_laser_off,
        target_x,
        target_y,
        bbox,
        all_bboxes,
        center_roi_box,
        area_box,
        response_u8,
        response_metrics,
    ):
        """Save Phase 3 raw images plus response-based overlays into the pointing log dir."""
        if not log_dir:
            return

        prefix = f"{log_dir}/iter_{base_iteration}_phase3_{phase3_iter}_{phase3_tag}"

        try:
            cv2.imwrite(f"{prefix}_led_on.jpg", img_led_on)
            cv2.imwrite(f"{prefix}_led_off.jpg", img_led_off)
            cv2.imwrite(f"{prefix}_laser_on.jpg", img_laser_on)
            cv2.imwrite(f"{prefix}_laser_off.jpg", img_laser_off)

            led_diff = cv2.absdiff(img_led_on, img_led_off)
            cv2.imwrite(f"{prefix}_led_diff.jpg", led_diff)
            cv2.imwrite(f"{prefix}_laser_diff_raw.jpg", response_u8)

            response_overlay = self._phase3_render_response_overlay(
                led_diff_bgr=led_diff,
                area_box=area_box,
                response_u8=response_u8,
                bbox=bbox,
                all_bboxes=all_bboxes,
                center_roi_box=center_roi_box,
                target_x=target_x,
                target_y=target_y,
                phase3_iter=phase3_iter,
                phase3_tag=phase3_tag,
                response_metrics=response_metrics,
            )
            cv2.imwrite(f"{prefix}_laser_diff_thresh.jpg", response_overlay)

            debug = img_laser_on.copy()
            x1, y1, x2, y2 = response_metrics.get("clipped_area_box", self._phase3_clip_box(area_box, debug.shape))
            tx = int(round(float(target_x)))
            ty = int(round(float(target_y)))
            if bbox:
                bx, by, bw, bh = [int(v) for v in bbox]
                cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
                cv2.putText(debug, "OBJECT", (bx, max(20, by - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            if all_bboxes:
                for mx, my, mw, mh in all_bboxes:
                    cv2.rectangle(debug, (int(mx), int(my)),
                                  (int(mx + mw), int(my + mh)), (100, 100, 100), 1)
            if center_roi_box:
                rx1, ry1, rx2, ry2 = [int(v) for v in center_roi_box]
                cv2.rectangle(debug, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
                cv2.putText(debug, self._get_center_roi_label(), (rx1, max(20, ry1 + 24)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            cv2.drawMarker(debug, (tx, ty), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 32, 2)
            cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 255, 255), 2)
            peak = response_metrics.get("peak_pos")
            if peak is not None:
                cv2.circle(debug, peak, 10, (0, 0, 255), 2)
                cv2.drawMarker(debug, peak, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(debug, f"Phase3 {phase3_iter}-{phase3_tag}", (20, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)
            cv2.putText(debug, f"Pan={float(getattr(self, '_curr_pan', 0.0)):.2f} Tilt={float(getattr(self, '_curr_tilt', 0.0)):.2f}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
            cv2.putText(
                debug,
                "Resp mean={:.1f} core={:.1f} max={:.1f}".format(
                    float(response_metrics.get("mean_delta", 0.0)),
                    float(response_metrics.get("core_delta", 0.0)),
                    float(response_metrics.get("max_delta", 0.0)),
                ),
                (20, 104),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imwrite(f"{prefix}_response_overlay.jpg", response_overlay)
            cv2.imwrite(f"{prefix}_debug.jpg", debug)

            # Pointing 탭 하단 패널은 Phase 3 디버그 전용으로 사용
            self.root.after(0, lambda img=response_overlay.copy(): self._show_laser_diff(img))
        except Exception as e:
            print(f"[Pointing] Phase3 debug save failed: {e}")
    
    def stop_aiming(self):
        """ """
        if hasattr(self, '_aiming_active') and self._aiming_active:
            self._aiming_active = False
            if hasattr(self, '_aiming_cancel_event'):
                self._aiming_cancel_event.set()
            if hasattr(self, '_pointing_img_event'):
                # _snap_and_wait  
                self._pointing_img_event.set()
            self.ctrl.send({"cmd": "laser", "value": 0})
            self.ctrl.send({"cmd": "led", "value": 0})
            print("[Pointing] Aiming stopped.")

    # ========== Helper Functions (Missing Re-added) ==========

    def _find_object_center(self, img_led_on, img_led_off, selection_roi_box=None):
        """   - Scan diff """
        if img_led_on is None or img_led_off is None:
            return None, None, None, None

        # 1) Scan: LED ON/OFF (diff)
        diff = cv2.absdiff(img_led_on, img_led_off)

        # 2) YOLO Detection diff
        results = []
        if hasattr(self, 'yolo') and self.yolo:
            #   ( )
            if not self.yolo._cached_model:
                if hasattr(self, 'scan_tab') and hasattr(self.scan_tab, 'yolo_weights'):
                    model_path = self.scan_tab.yolo_weights.get()
                    if model_path:
                        self.yolo.get_model(model_path)

            # Scan  
            results = self.yolo.detect(diff, conf=0.20, iou=0.45)
        else:
            results = []

        all_bboxes = []
        target_bbox = None
        target_center = None
        self._last_object_led_info = {"pred": "NONE", "score": {"R": 0, "G": 0, "B": 0}, "roi": None}

        H, W = diff.shape[:2]
        center_x, center_y = W // 2, H // 2

        # 3) conf>=0.5    ( )
        use_results = [r for r in results if len(r) >= 6 and float(r[4]) >= 0.5] or results
        led_params = getattr(self, "led_filter_params", None) or get_default_led_filter_params()
        candidates = []

        # 4)   + LED   
        for r in use_results:
            x1, y1, x2, y2, conf, cls_id = r
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2

            bbox = (int(x1), int(y1), int(w), int(h))
            all_bboxes.append(bbox)
            dist = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5
            # LED 판정은 항상 LED OFF 단일 프레임 기준으로 수행
            led_roi_seed = expand_led_roi_from_bbox(
                bbox,
                img_led_off.shape,
                top_ratio=1.0 / 3.0,
            )
            led_pred, led_score, led_roi = classify_from_single_roi(
                img_led_off,
                led_roi_seed,
                params=led_params,
            )
            led_strength = max(int(led_score["R"]), int(led_score["G"]), int(led_score["B"]))
            candidates.append({
                "bbox": bbox,
                "center": (cx, cy),
                "dist": dist,
                "in_selection_roi": self._point_in_box(cx, cy, selection_roi_box),
                "led_pred": led_pred,
                "led_score": led_score,
                "led_roi": led_roi,
                "led_strength": led_strength,
            })

        # 4-1)  :    
        if candidates:
            selectable = candidates
            if selection_roi_box is not None:
                selectable = [c for c in candidates if c.get("in_selection_roi", False)]
            if selectable:
                selectable.sort(key=lambda c: c["dist"])
                best = selectable[0]

                target_bbox = best["bbox"]
                target_center = best["center"]
                self._last_object_led_info = {
                    "pred": best["led_pred"],
                    "score": dict(best["led_score"]),
                    "roi": best["led_roi"],
                }

        # 5) YOLO   diff blob  fallback ( 
        if target_center is None:
            try:
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                #  20~50  
                _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
                mask = cv2.medianBlur(mask, 5)

                fc = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = fc[0] if len(fc) == 2 else fc[1]

                if cnts:
                    c = max(cnts, key=cv2.contourArea)
                    if cv2.contourArea(c) > 30:  #  
                        x, y, w, h = cv2.boundingRect(c)
                        cx = x + w / 2.0
                        cy = y + h / 2.0
                        all_bboxes = [(int(x), int(y), int(w), int(h))]
                        self._last_object_led_info = {"pred": "NONE", "score": {"R": 0, "G": 0, "B": 0}, "roi": None}
                        if selection_roi_box is not None and not self._point_in_box(cx, cy, selection_roi_box):
                            return None, None, None, all_bboxes
                        return cx, cy, (int(x), int(y), int(w), int(h)), all_bboxes
            except Exception:
                pass

            print("[Pointing]  YOLO(diff): ")
            return None, None, None, all_bboxes

        return target_center[0], target_center[1], target_bbox, all_bboxes

    def _find_laser_center(self, img_on, img_off, exclude_bboxes=None):
        """Laser ON/OFF diff 기반 레이저 중심 추정 (blob 미사용, THRESH_TOZERO+moments)."""
        return self._find_laser_center_with_roi(
            img_on=img_on,
            img_off=img_off,
            exclude_bboxes=exclude_bboxes,
            roi_center=None,
            roi_half_size=300,
            tozero_threshold=PHASE3_DIFF_TOZERO_THRESH,
        )

    def _find_laser_center_with_roi(
        self,
        img_on,
        img_off,
        exclude_bboxes=None,
        roi_center=None,
        roi_half_size=300,
        tozero_threshold=70.0,
    ):
        """Laser ON/OFF diff에서 지정 ROI 중심의 레이저 좌표를 moments로 계산."""
        if img_on is None or img_off is None:
            return None
            
        H, W = img_on.shape[:2]

        if roi_center is None:
            cy, cx = H // 2, W // 2
            roi_y1 = max(0, cy - roi_half_size - 100)
            roi_y2 = min(H, cy + roi_half_size - 100)
            roi_x1 = max(0, cx - roi_half_size)
            roi_x2 = min(W, cx + roi_half_size)
        else:
            cx = int(round(float(roi_center[0])))
            cy = int(round(float(roi_center[1])))
            roi_y1 = max(0, cy - roi_half_size)
            roi_y2 = min(H, cy + roi_half_size)
            roi_x1 = max(0, cx - roi_half_size)
            roi_x2 = min(W, cx + roi_half_size)
        if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
            return None
        
        roi_on = img_on[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_off = img_off[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Diff + THRESH_TOZERO: 임계값 이하 제거 (view_diff/diff_laser 방식)
        diff = cv2.absdiff(roi_on, roi_off)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, weight = cv2.threshold(
            gray,
            float(tozero_threshold),
            255,
            cv2.THRESH_TOZERO,
        )
        weight = weight.astype(np.float32)
        
        # (  )
        if exclude_bboxes:
            for (bx, by, bw, bh) in exclude_bboxes:
                rx1 = max(0, bx - roi_x1)
                ry1 = max(0, by - roi_y1)
                rx2 = min(roi_x2 - roi_x1, bx + bw - roi_x1)
                ry2 = min(roi_y2 - roi_y1, by + bh - roi_y1)
                
                if rx1 < rx2 and ry1 < ry2:
                    weight[ry1:ry2, rx1:rx2] = 0.0

        #   (Intensity-weighted centroid)
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

    def _phase3_refine_laser_target(
        self,
        track_id,
        target_x,
        target_y,
        all_bboxes,
        k_pan,
        k_tilt,
        settle,
        led_settle,
        log_dir,
        base_iteration,
        initial_px_per_cm=None,
    ):
        """Phase2 완료 후 Phase3: pan 3점(현재, -1, +1)에서 area response 최대 후보 선택."""
        px_per_cm_hint = float(initial_px_per_cm) if initial_px_per_cm else None
        last_bboxes = all_bboxes
        last_target_x = float(target_x)
        last_target_y = float(target_y)
        base_pan = float(self._curr_pan)
        base_tilt = float(self._curr_tilt)
        candidates = []

        def _measure_at_pan(pan_value, phase3_iter, tag):
            nonlocal px_per_cm_hint, last_bboxes, last_target_x, last_target_y
            self._curr_pan = float(pan_value)
            self._curr_tilt = float(base_tilt)
            self.ctrl.send(
                {
                    "cmd": "move",
                    "pan": self._curr_pan,
                    "tilt": self._curr_tilt,
                    "speed": 100,
                    "acc": 1.0,
                }
            )
            time.sleep(settle)

            meas, px_per_cm_hint, last_bboxes = self._phase3_measure_error(
                track_id=track_id,
                base_iteration=base_iteration,
                phase3_iter=phase3_iter,
                phase3_tag=tag,
                log_dir=log_dir,
                led_settle=led_settle,
                fallback_target=(last_target_x, last_target_y),
                fallback_bboxes=last_bboxes,
                px_per_cm_hint=px_per_cm_hint,
            )
            if meas is not None:
                last_target_x, last_target_y = meas["target_x"], meas["target_y"]
            return meas

        # 1) pan +1
        pan_plus = float(max(-180.0, min(180.0, base_pan + PHASE3_STEP_DEG)))
        if abs(pan_plus - base_pan) > 1e-6:
            meas_plus = _measure_at_pan(pan_plus, 1, "plus")
            if meas_plus is not None:
                candidates.append(meas_plus)
                print(
                    f"[Pointing-Adaptive] Phase 3-3pt plus: pan={meas_plus['pan']:.2f}, "
                    f"mean={meas_plus['response_mean']:.1f}, core={meas_plus['response_core']:.1f}, "
                    f"max={meas_plus['response_max']:.1f}"
                )

        # 2) base
        meas_base = _measure_at_pan(base_pan, 2, "base")
        if meas_base is not None:
            candidates.append(meas_base)
            print(
                f"[Pointing-Adaptive] Phase 3-3pt base: pan={meas_base['pan']:.2f}, "
                f"mean={meas_base['response_mean']:.1f}, core={meas_base['response_core']:.1f}, "
                f"max={meas_base['response_max']:.1f}"
            )

        # 3) pan -1
        pan_minus = float(max(-180.0, min(180.0, base_pan - PHASE3_STEP_DEG)))
        if abs(pan_minus - base_pan) > 1e-6:
            meas_minus = _measure_at_pan(pan_minus, 3, "minus")
            if meas_minus is not None:
                candidates.append(meas_minus)
                print(
                    f"[Pointing-Adaptive] Phase 3-3pt minus: pan={meas_minus['pan']:.2f}, "
                    f"mean={meas_minus['response_mean']:.1f}, core={meas_minus['response_core']:.1f}, "
                    f"max={meas_minus['response_max']:.1f}"
                )

        if not candidates:
            self._curr_pan = base_pan
            self._curr_tilt = base_tilt
            self.ctrl.send(
                {
                    "cmd": "move",
                    "pan": self._curr_pan,
                    "tilt": self._curr_tilt,
                    "speed": 100,
                    "acc": 1.0,
                }
            )
            self.ctrl.send({"cmd": "laser", "value": 1})
            return False, None

        # area 평균 반응 최대 우선, 동률이면 core/max 반응 순으로 선택
        best_meas = max(
            candidates,
            key=lambda m: (
                float(m["response_mean"]),
                float(m["response_core"]),
                float(m["response_max"]),
            ),
        )

        self._curr_pan = float(best_meas["pan"])
        self._curr_tilt = float(base_tilt)
        self.ctrl.send(
            {
                "cmd": "move",
                "pan": self._curr_pan,
                "tilt": self._curr_tilt,
                "speed": 100,
                "acc": 1.0,
            }
        )
        best_led_roi = best_meas.get("led_roi")
        best_led_roi_source_size = best_meas.get("led_roi_source_size")
        if best_led_roi is not None and hasattr(self, "_track_led_roi"):
            try:
                self._track_led_roi[track_id] = tuple(int(v) for v in best_led_roi)
                if hasattr(self, "_track_led_roi_source_size") and best_led_roi_source_size is not None:
                    self._track_led_roi_source_size[track_id] = (
                        int(best_led_roi_source_size[0]),
                        int(best_led_roi_source_size[1]),
                    )
            except Exception:
                pass
        pan_q, tilt_q = self._quantize_pan_tilt(self._curr_pan, self._curr_tilt)
        self.computed_targets[track_id] = (pan_q, tilt_q)
        self._update_target_button_value(track_id, pan_q, tilt_q)

        print(
            f"[Pointing-Adaptive] Phase 3-3pt best: pan={self._curr_pan:.2f}, "
            f"mean={best_meas['response_mean']:.1f}, core={best_meas['response_core']:.1f}, "
            f"max={best_meas['response_max']:.1f}"
        )
        self.ctrl.send({"cmd": "laser", "value": 1})
        return True, (
            float(best_meas["pan"]),
            float(best_meas["tilt"]),
            float(best_meas["response_mean"]),
            float(best_meas["response_core"]),
            float(best_meas["response_max"]),
        )

    def _phase3_measure_error(
        self,
        track_id,
        base_iteration,
        phase3_iter,
        phase3_tag,
        log_dir,
        led_settle,
        fallback_target,
        fallback_bboxes,
        px_per_cm_hint,
    ):
        """Phase3 단일 측정: 타겟 재검출 + area response 계산."""
        try:
            self.set_ir_cut("night")
            time.sleep(0.05)

            self.ctrl.send({"cmd": "led", "value": 255})
            time.sleep(led_settle)
            img_led_on = self._snap_and_wait(
                f"pointing_phase3_led_on_{base_iteration}_{phase3_iter}_{phase3_tag}",
            )
            self.ctrl.send({"cmd": "led", "value": 0})
            if img_led_on is None:
                self.set_ir_cut("day")
                return None, px_per_cm_hint, fallback_bboxes

            time.sleep(led_settle)
            img_led_off = self._snap_and_wait(
                f"pointing_phase3_led_off_{base_iteration}_{phase3_iter}_{phase3_tag}",
            )
            self.set_ir_cut("day")
            if img_led_off is None:
                return None, px_per_cm_hint, fallback_bboxes

            center_roi_box = self._get_center_roi_box(img_led_on.shape)
            obj_cx, obj_cy, bbox, all_bboxes = self._find_object_center(
                img_led_on,
                img_led_off,
                selection_roi_box=center_roi_box,
            )
            if all_bboxes is None:
                all_bboxes = fallback_bboxes

            if obj_cx is None and not px_per_cm_hint:
                px_per_cm_hint = 10.0
            if bbox is None:
                print(
                    "[Pointing-Adaptive] Phase 3: no object inside center ROI x "
                    f"{PHASE23_CENTER_ROI_SIZE_PX}px"
                )
                return None, px_per_cm_hint, all_bboxes

            led_roi = getattr(self, "_last_object_led_info", {}).get("roi")
            led_roi_source_size = (int(img_led_on.shape[1]), int(img_led_on.shape[0]))

            geometry = self._phase3_build_area_geometry(
                bbox=bbox,
                fallback_target=fallback_target,
                px_per_cm_hint=px_per_cm_hint,
            )
            target_x = geometry["target_x"]
            target_y = geometry["target_y"]
            px_per_cm_hint = geometry["px_per_cm"]
            area_box = geometry["area_box"]

            self.ctrl.send({"cmd": "laser", "value": 1})
            time.sleep(led_settle)
            img_laser_on = self._snap_and_wait(
                f"pointing_phase3_laser_on_{base_iteration}_{phase3_iter}_{phase3_tag}",
            )
            self.ctrl.send({"cmd": "laser", "value": 0})
            if img_laser_on is None:
                return None, px_per_cm_hint, all_bboxes

            time.sleep(led_settle)
            img_laser_off = self._snap_and_wait(
                f"pointing_phase3_laser_off_{base_iteration}_{phase3_iter}_{phase3_tag}",
            )
            if img_laser_off is None:
                return None, px_per_cm_hint, all_bboxes

            response_u8, response_metrics = self._phase3_compute_response_metrics(
                img_laser_on=img_laser_on,
                img_laser_off=img_laser_off,
                area_box=area_box,
            )

            self._save_phase3_debug_artifacts(
                log_dir=log_dir,
                base_iteration=base_iteration,
                phase3_iter=phase3_iter,
                phase3_tag=phase3_tag,
                img_led_on=img_led_on,
                img_led_off=img_led_off,
                img_laser_on=img_laser_on,
                img_laser_off=img_laser_off,
                target_x=target_x,
                target_y=target_y,
                bbox=bbox,
                all_bboxes=all_bboxes,
                center_roi_box=center_roi_box,
                area_box=area_box,
                response_u8=response_u8,
                response_metrics=response_metrics,
            )

            try:
                with open(f"{log_dir}/log.txt", "a", encoding="utf-8") as f:
                    f.write(
                        "Iter {it} Phase3-{p3}-{tag}: Target=({tx:.1f},{ty:.1f}) "
                        "RespMean={rm:.2f} RespCore={rc:.2f} RespMax={rx:.2f} Pose=({cp:.3f},{ct:.3f})\n".format(
                            it=base_iteration,
                            p3=phase3_iter,
                            tag=phase3_tag,
                            tx=target_x,
                            ty=target_y,
                            rm=float(response_metrics.get("mean_delta", 0.0)),
                            rc=float(response_metrics.get("core_delta", 0.0)),
                            rx=float(response_metrics.get("max_delta", 0.0)),
                            cp=float(self._curr_pan),
                            ct=float(self._curr_tilt),
                        )
                    )
            except Exception as e:
                print(f"[Pointing-Adaptive] Log write failed: {e}")

            return {
                "target_x": float(target_x),
                "target_y": float(target_y),
                "response_mean": float(response_metrics.get("mean_delta", 0.0)),
                "response_core": float(response_metrics.get("core_delta", 0.0)),
                "response_max": float(response_metrics.get("max_delta", 0.0)),
                "pan": float(self._curr_pan),
                "tilt": float(self._curr_tilt),
                "led_roi": tuple(int(v) for v in led_roi) if led_roi is not None else None,
                "led_roi_source_size": led_roi_source_size,
            }, px_per_cm_hint, all_bboxes
        except Exception as e:
            print(f"[Pointing-Adaptive] Phase 3 measure failed: {e}")
            return None, px_per_cm_hint, fallback_bboxes

    def _phase3_get_roi_half_size(self, aim_x, aim_y, bboxes):
        """조준점 중심 ROI가 타겟 bbox에 닿지 않도록 half-size를 자동 조정."""
        default_half = int(PHASE3_ROI_HALF_SIZE_PX)
        if not bboxes:
            return default_half

        ax = float(aim_x)
        ay = float(aim_y)
        margin = int(PHASE3_ROI_MARGIN_FROM_TARGET_PX)
        max_allowed = float("inf")

        for bx, by, bw, bh in bboxes:
            x1 = float(bx)
            y1 = float(by)
            x2 = float(bx + bw)
            y2 = float(by + bh)

            # 조준점이 bbox 아래/위에 있을 때 수직 간격 기반으로 ROI 상하 경계 제한
            if ay >= y2:
                gap = ay - y2
            elif ay <= y1:
                gap = y1 - ay
            else:
                # 조준점 y가 bbox 범위에 겹치면 매우 작은 ROI만 허용
                gap = 0.0

            allowed = gap - margin
            if allowed < max_allowed:
                max_allowed = allowed

        if max_allowed == float("inf"):
            return default_half

        if max_allowed <= 8:
            return 8
        if max_allowed < default_half:
            return int(max_allowed)
        return default_half

    def _calculate_angle_delta(self, err_x, err_y, k_pan, k_tilt):
        """Convert pixel error to pan/tilt step with max-step clamp."""
        d_pan = err_x * k_pan
        d_tilt = -err_y * k_tilt
        d_pan = max(min(d_pan, MAX_STEP_DEG), -MAX_STEP_DEG)
        d_tilt = max(min(d_tilt, MAX_STEP_DEG), -MAX_STEP_DEG)
        return d_pan, d_tilt




