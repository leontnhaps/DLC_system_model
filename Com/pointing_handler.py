"""
Pointing mode handler mixin
Handles CSV analysis and target computation
"""

import csv
import numpy as np
from tkinter import filedialog
from collections import defaultdict


class PointingHandlerMixin:
    """Pointing mode logic - CSV analysis and regression"""
    
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
                        "a": float(a),
                        "b": float(b),
                        "R2": float(R2),
                        "N": int(len(arr)),
                        "pan_center": float(pan_center),
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
                        "e": float(e),
                        "f": float(f),
                        "R2": float(R2),
                        "N": int(len(arr)),
                        "tilt_center": float(tilt_center),
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
                
                # Track ID별 결과 저장
                if pan_target is not None and tilt_target is not None:
                    self.computed_targets[track_id] = (round(pan_target, 3), round(tilt_target, 3))
                    print(f"[Pointing] track_id={track_id} → pan={pan_target:.3f}°, tilt={tilt_target:.3f}° "
                          f"(H fits: {len(fits_h)}, V fits: {len(fits_v)})")
                else:
                    print(f"[Pointing] track_id={track_id} → 계산 실패 (insufficient data)")
            
            # UI 업데이트 (버튼 생성)
            if self.computed_targets:
                print(f"[Pointing] {len(self.computed_targets)} target(s) computed")
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
    
    def move_to_target(self, track_id):
        """
        특정 track_id의 계산된 pan/tilt로 카메라 이동
        
        Args:
            track_id: 이동할 track의 ID
        """
        if not hasattr(self, 'computed_targets') or track_id not in self.computed_targets:
            print(f"[Pointing] Track {track_id} 타깃 없음. 먼저 계산하세요")
            return
        
        pan_t, tilt_t = self.computed_targets[track_id]
        spd = int(100)
        acc = float(1.0)
        
        # 이동
        self.ctrl.send({"cmd": "move", "pan": pan_t, "tilt": tilt_t, "speed": spd, "acc": acc})
        print(f"[Pointing] Track {track_id}: Move to (pan={pan_t}°, tilt={tilt_t}°)")
