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


class PointingHandlerMixin:
    """Pointing mode logic - CSV analysis, regression, and laser fine-aiming"""

    @staticmethod
    def _quantize_deg(value):
        """Quantize servo command to integer degree (round)."""
        return float(int(round(float(value))))

    def _quantize_pan_tilt(self, pan, tilt):
        return self._quantize_deg(pan), self._quantize_deg(tilt)
    
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
            
            # CSV 
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
                    
                    # Track ID 
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
                
                # Track ID 
                if pan_target is not None and tilt_target is not None:
                    pan_q, tilt_q = self._quantize_pan_tilt(pan_target, tilt_target)
                    self.computed_targets[track_id] = (pan_q, tilt_q)
                    self._pointing_gains[track_id] = (k_pan, k_tilt)
                    print(f"[Pointing] track_id={track_id} pan={pan_q:.3f}, tilt={tilt_q:.3f} "
                          f"(H fits: {len(fits_h)}, V fits: {len(fits_v)}, gain: k_p={k_pan:.5f}, k_t={k_tilt:.5f})")
                else:
                    print(f"[Pointing] track_id={track_id}   (insufficient data)")
            
            #  : 5   ID
            MERGE_TOL = 5.0  # deg
            merged = self._merge_similar_targets(self.computed_targets, grouped_by_track, MERGE_TOL, W_frame, H_frame, min_samples)
            if merged:
                self.computed_targets = merged['targets']
                self._pointing_gains = merged['gains']

            # TrackLED ROI (CSVled_roi_*  
            #   track_id , trackROI 
            track_led_roi = {}
            for tid in self.computed_targets.keys():
                samples = track_led_roi_samples.get(tid, [])
                if not samples:
                    continue
                arr = np.array(samples, dtype=float)
                med = np.median(arr, axis=0)
                roi = tuple(int(v) for v in med.tolist())
                if roi[2] > 0 and roi[3] > 0:
                    track_led_roi[int(tid)] = roi
            self._track_led_roi = track_led_roi
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
        new_rois = {}

        for old_id in old_ids:
            new_id = id_map[old_id]
            new_targets[new_id] = self.computed_targets[old_id]
            gain = self._pointing_gains.get(old_id, (CENTERING_GAIN_PAN, CENTERING_GAIN_TILT))
            new_gains[new_id] = gain
            if old_id in old_rois:
                new_rois[new_id] = old_rois[old_id]

        self.computed_targets = new_targets
        self._pointing_gains = new_gains
        self._track_led_roi = new_rois
        print(f"[Pointing] ID renumbered: {id_map}")
    
    def _merge_similar_targets(self, targets, grouped_by_track, tol, W_frame, H_frame, min_samples):
        """
        Merge nearby targets within tol and recompute representative target.
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
        
        # (  ) None 
        if all(len(g) == 1 for g in merged_groups):
            return None
        
        print(f"[Pointing]    (tol={tol}):")
        new_targets = {}
        new_gains = {}
        
        for group in merged_groups:
            rep_id = min(group)  #  ID 
            
            if len(group) == 1:
                #  
                new_targets[rep_id] = targets[rep_id]
                new_gains[rep_id] = self._pointing_gains.get(rep_id, (CENTERING_GAIN_PAN, CENTERING_GAIN_TILT))
                continue
            
            print(f"  IDs {group} ID {rep_id}  ")
            
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
        
        return {'targets': new_targets, 'gains': new_gains}
    
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
    
    def move_to_target(self, track_id, use_tilt_approach=False):
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

        #   ( tilt+1 ->  tilt )
        if use_tilt_approach:
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
                       .
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

                obj_cx, obj_cy, bbox, all_bboxes = self._find_object_center(img_led_on, img_led_off)
                if obj_cx is None:
                    print("[Pointing-Adaptive]   ")
                    continue
                led_roi = getattr(self, "_last_object_led_info", {}).get("roi")
                if led_roi is not None and hasattr(self, "_track_led_roi"):
                    try:
                        self._track_led_roi[track_id] = tuple(int(v) for v in led_roi)
                    except Exception:
                        pass

                H, W = img_led_on.shape[:2]
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
                        self.ctrl.send({"cmd": "laser", "value": 1})
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
                            tol_x=None, tol_y=None):
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
            print(f"[Pointing] Debug preview : {e}")
    
    def _show_laser_diff(self, img_bgr):
        """Laser diff Pointing   (main thread)"""
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

    def _find_object_center(self, img_led_on, img_led_off):
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
                "led_pred": led_pred,
                "led_score": led_score,
                "led_roi": led_roi,
                "led_strength": led_strength,
            })

        # 4-1)  :    
        if candidates:
            candidates.sort(key=lambda c: c["dist"])
            best = candidates[0]

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
                        return cx, cy, (int(x), int(y), int(w), int(h)), all_bboxes
            except Exception:
                pass

            print("[Pointing]  YOLO(diff): ")
            return None, None, None, all_bboxes

        return target_center[0], target_center[1], target_bbox, all_bboxes

    def _find_laser_center(self, img_on, img_off, exclude_bboxes=None):
        """  """
        if img_on is None or img_off is None:
            return None
            
        H, W = img_on.shape[:2]
        
        # ROI 
        crop_size = 300
        cy, cx = H // 2, W // 2
        roi_y1 = max(0, cy - crop_size - 100) 
        roi_y2 = min(H, cy + crop_size - 100)
        roi_x1 = max(0, cx - crop_size)
        roi_x2 = min(W, cx + crop_size)
        
        roi_on = img_on[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_off = img_off[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Diff & band-pass threshold (100~200 )
        diff = cv2.absdiff(roi_on, roi_off)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        thr_low = 100.0
        thr_high = 150.0
        weight = gray.astype(np.float32)
        weight[(weight < thr_low) | (weight > thr_high)] = 0.0
        
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

    def _calculate_angle_delta(self, err_x, err_y, k_pan, k_tilt):
        """Convert pixel error to pan/tilt step with max-step clamp."""
        d_pan = err_x * k_pan
        d_tilt = -err_y * k_tilt
        d_pan = max(min(d_pan, MAX_STEP_DEG), -MAX_STEP_DEG)
        d_tilt = max(min(d_tilt, MAX_STEP_DEG), -MAX_STEP_DEG)
        return d_pan, d_tilt




