"""
MOT 알고리즘 비교 실험 스크립트
3가지 버전 비교:
1. Original: Grid 11x11 + 5개 프레임 타입 후보 (공간 기반 후보 선택)
2. No Grid: 전체 ROI 히스토그램 + 5개 프레임 타입 후보
3. Temporal n=5: Grid 11x11 + 시간 순서로 최근 5개 프레임의 모든 객체를 후보로
"""
import cv2
import numpy as np
import sys
import os
import re
from pathlib import Path
from numpy.linalg import norm
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
import json

# ---------------------------------------------------------
# 기존 모듈 로드
# ---------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
detection_test_dir = os.path.join(script_dir, '..', 'Detection_test')
sys.path.insert(0, detection_test_dir)

try:
    from yolo_utils import predict_with_tiling
    print("✅ yolo_utils 로드 성공!")
except ImportError as e:
    print(f"❌ 오류: yolo_utils.py를 찾을 수 없습니다: {e}")
    sys.exit()

# =========================================================
# [설정] 스캔 이미지 폴더 경로
# =========================================================
MODEL_PATH = "yolov11m_diff.pt"
SCAN_FOLDER = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon2학년\2026_통신학회동계\captures_gui_20260107_181822"

CONF_THRES = 0.50
IOU_THRES = 0.45
ROI_SIZE = 300  # 300x300 픽셀

# =========================================================
# 특징 추출 함수
# =========================================================

def get_feature_vector_grid(roi_bgr, diff_roi=None, grid_size=(11, 11)):
    """Grid 11x11 히스토그램 특징"""
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = hsv.shape[:2]
    rows, cols = grid_size
    
    feature_vector = []
    
    for r in range(rows):
        for c in range(cols):
            y_start = int((r / rows) * h)
            y_end = int(((r + 1) / rows) * h)
            x_start = int((c / cols) * w)
            x_end = int(((c + 1) / cols) * w)
            
            cell_hsv = hsv[y_start:y_end, x_start:x_end]
            cell_gray = gray[y_start:y_end, x_start:x_end]
            
            if diff_roi is not None:
                diff_cell = diff_roi[y_start:y_end, x_start:x_end]
                if len(diff_cell.shape) == 3:
                    diff_gray = cv2.cvtColor(diff_cell, cv2.COLOR_BGR2GRAY)
                else:
                    diff_gray = diff_cell
                
                diff_mask = (diff_gray < 20).astype(np.uint8) * 255
                v_mask = cv2.inRange(cell_hsv, (0, 0, 30), (180, 255, 255))
                mask = cv2.bitwise_and(diff_mask, v_mask)
            else:
                mask = cv2.inRange(cell_hsv, (0, 0, 30), (180, 255, 255))
            
            hist_hsv = cv2.calcHist([cell_hsv], [0, 1], mask, [8, 4], [0, 180, 0, 256])
            cv2.normalize(hist_hsv, hist_hsv, 0, 1, cv2.NORM_MINMAX)
            
            hist_gray = cv2.calcHist([cell_gray], [0], mask, [16], [0, 256])
            cv2.normalize(hist_gray, hist_gray, 0, 1, cv2.NORM_MINMAX)
            
            combined_hist = np.concatenate([hist_hsv.flatten(), hist_gray.flatten()])
            feature_vector.append(combined_hist)
    
    final_vector = np.concatenate(feature_vector)
    final_vector = final_vector / (norm(final_vector) + 1e-7)
    
    return final_vector


def get_feature_vector_no_grid(roi_bgr, diff_roi=None):
    """전체 ROI 히스토그램 특징 (Grid 없음)"""
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    
    if diff_roi is not None:
        if len(diff_roi.shape) == 3:
            diff_gray = cv2.cvtColor(diff_roi, cv2.COLOR_BGR2GRAY)
        else:
            diff_gray = diff_roi
        
        diff_mask = (diff_gray < 20).astype(np.uint8) * 255
        v_mask = cv2.inRange(hsv, (0, 0, 30), (180, 255, 255))
        mask = cv2.bitwise_and(diff_mask, v_mask)
    else:
        mask = cv2.inRange(hsv, (0, 0, 30), (180, 255, 255))
    
    # 더 세밀한 히스토그램 (128 + 32 = 160차원)
    hist_hsv = cv2.calcHist([hsv], [0, 1], mask, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist_hsv, hist_hsv, 0, 1, cv2.NORM_MINMAX)
    
    hist_gray = cv2.calcHist([gray], [0], mask, [32], [0, 256])
    cv2.normalize(hist_gray, hist_gray, 0, 1, cv2.NORM_MINMAX)
    
    final_vector = np.concatenate([hist_hsv.flatten(), hist_gray.flatten()])
    final_vector = final_vector / (norm(final_vector) + 1e-7)
    
    return final_vector


def calc_cosine_similarity(vec_a, vec_b):
    """코사인 유사도"""
    if vec_a is None or vec_b is None:
        return 0.0
    dot = np.dot(vec_a, vec_b)
    n_a, n_b = norm(vec_a), norm(vec_b)
    if n_a == 0 or n_b == 0:
        return 0.0
    return dot / (n_a * n_b)


# =========================================================
# ObjectTracker 클래스 (3가지 버전)
# =========================================================

class ObjectTrackerBase:
    """기본 Tracker"""
    def __init__(self, version_name):
        self.version_name = version_name
        self.next_id = 0
        self.frames = []
        self.unique_id_counter = 1
        
    def reset(self):
        self.next_id = 0
        self.frames = []
        self.unique_id_counter = 1
    
    @property
    def frame_objects(self):
        """시각화용"""
        result = {}
        for frame in self.frames:
            key = (frame['pan'], frame['tilt'])
            result[key] = frame['objects']
        return result
    
    def merge_similar_tracks(self, merge_threshold=0.4, min_detections=3):
        """유사한 Track들을 병합 (hungarian_final과 동일)"""
        print(f"\n{'='*60}")
        print(f"🔄 Track 병합 시작 (threshold={merge_threshold}, min={min_detections})")
        print(f"{'='*60}")
        
        # 1. 각 track별 검출 수집
        tracks = {}
        for frame in self.frames:
            for obj in frame['objects']:
                track_id = obj['track_id']
                if track_id not in tracks:
                    tracks[track_id] = []
                tracks[track_id].append(obj)
        
        # 2. 모든 Track 처리
        all_track_ids = list(tracks.keys())
        valid_tracks = {tid: objs for tid, objs in tracks.items() if len(objs) >= min_detections}
        small_tracks = {tid: objs for tid, objs in tracks.items() if 1 <= len(objs) < min_detections}
        
        print(f"  총 Track 수: {len(tracks)}개")
        print(f"  큰 Track (>= {min_detections}개): {len(valid_tracks)}개")
        print(f"  작은 Track (1~{min_detections-1}개): {len(small_tracks)}개")
        
        # 3. Track별 위치 정보 수집
        track_by_position = {}
        for frame in self.frames:
            pan = frame['pan']
            tilt = frame['tilt']
            for obj in frame['objects']:
                tid = obj['track_id']
                if tid not in track_by_position:
                    track_by_position[tid] = []
                track_by_position[tid].append((pan, tilt, obj))
        
        # 4. Track 간 유사도 계산 및 병합
        merge_groups = []
        visited = set()
        
        for i, tid_a in enumerate(all_track_ids):
            if tid_a in visited:
                continue
            
            group = [tid_a]
            visited.add(tid_a)
            positions_a = track_by_position.get(tid_a, [])
            
            for j in range(i + 1, len(all_track_ids)):
                tid_b = all_track_ids[j]
                if tid_b in visited:
                    continue
                
                positions_b = track_by_position.get(tid_b, [])
                
                # 공간 기반 샘플링: 같은 Pan 라인 또는 같은 Tilt 라인만 비교
                similarities = []
                
                # 같은 Pan 라인
                for pan_a, tilt_a, obj_a in positions_a:
                    for pan_b, tilt_b, obj_b in positions_b:
                        if pan_a == pan_b:
                            sim = calc_cosine_similarity(obj_a['vec'], obj_b['vec'])
                            similarities.append(sim)
                
                # 같은 Tilt 라인
                for pan_a, tilt_a, obj_a in positions_a:
                    for pan_b, tilt_b, obj_b in positions_b:
                        if tilt_a == tilt_b:
                            sim = calc_cosine_similarity(obj_a['vec'], obj_b['vec'])
                            similarities.append(sim)
                
                if not similarities:
                    continue
                
                avg_sim = np.mean(similarities)
                
                if avg_sim >= merge_threshold:
                    print(f"  ✅ Track {tid_a}({len(positions_a)}개) ↔ Track {tid_b}({len(positions_b)}개): "
                          f"유사도 {avg_sim:.4f} → 병합!")
                    group.append(tid_b)
                    visited.add(tid_b)
            
            merge_groups.append(group)
        
        # 5. 병합 맵 생성
        merge_map = {}
        for group in merge_groups:
            representative_id = min(group)
            for tid in group:
                merge_map[tid] = representative_id
        
        # 6. 모든 프레임의 track_id 업데이트
        merged_count = 0
        for frame in self.frames:
            for obj in frame['objects']:
                old_id = obj['track_id']
                if old_id in merge_map:
                    new_id = merge_map[old_id]
                    if old_id != new_id:
                        merged_count += 1
                    obj['track_id'] = new_id
        
        print(f"\n  병합 결과:")
        print(f"    최종 Track 수: {len(merge_groups)}개")
        print(f"    병합된 검출: {merged_count}개")
        print(f"{'='*60}\n")
        
        return merge_map


class ObjectTrackerOriginal(ObjectTrackerBase):
    """버전 1: Grid 11x11 + 5개 프레임 타입 후보 (공간 기반)"""
    def __init__(self):
        super().__init__("V1_Grid_Spatial")
        
    def add_detections(self, boxes, scores, img_on, diff, pan, tilt, timestamp):
        curr_objects = []
        H, W = img_on.shape[:2]
        
        for i, (x, y, w, h) in enumerate(boxes):
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            
            half_size = ROI_SIZE // 2
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(W, center_x + half_size)
            y2 = min(H, center_y + half_size)
            
            roi = img_on[y1:y2, x1:x2]
            diff_roi = diff[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
                
            vec = get_feature_vector_grid(roi, diff_roi=diff_roi, grid_size=(11, 11))
            
            unique_id = str(self.unique_id_counter)
            self.unique_id_counter += 1
            
            curr_objects.append({
                'box': (x, y, w, h),
                'vec': vec,
                'idx': i,
                'unique_id': unique_id,
                'roi_img': roi.copy()
            })
        
        # 5개 프레임 타입 후보 (공간 기반)
        candidates_dict = self._find_spatial_candidates(pan, tilt)
        all_candidates = candidates_dict['direct'] + candidates_dict['skip']
        
        track_ids = self._hungarian_matching(curr_objects, all_candidates)
        
        self.frames.append({
            'pan': pan,
            'tilt': tilt,
            'timestamp': timestamp,
            'objects': curr_objects
        })
        
        return track_ids
    
    def _find_spatial_candidates(self, current_pan, current_tilt):
        """5개 프레임 타입 (공간 기반 선택)"""
        if not self.frames:
            return {'direct': [], 'skip': []}
        
        prev_pan_frame = None
        prev_tilt_frame = None
        skip_pan_frame = None
        diagonal_increase = None
        diagonal_decrease = None
        
        for i in range(len(self.frames)):
            prev_frame = self.frames[-(i+1)]
            frame_pan = prev_frame['pan']
            frame_tilt = prev_frame['tilt']
            
            if i == 0:  # n-1
                if frame_pan == current_pan and frame_tilt != current_tilt and prev_pan_frame is None:
                    prev_pan_frame = prev_frame
                if frame_tilt == current_tilt and frame_pan != current_pan and prev_tilt_frame is None:
                    prev_tilt_frame = prev_frame
            
            elif i == 1:  # n-2
                if frame_pan == current_pan and frame_tilt != current_tilt and skip_pan_frame is None:
                    skip_pan_frame = prev_frame
            
            # 양방향 대각선
            if frame_pan != current_pan and frame_tilt != current_tilt:
                if frame_pan < current_pan and diagonal_increase is None:
                    diagonal_increase = prev_frame
                if frame_pan > current_pan and diagonal_decrease is None:
                    diagonal_decrease = prev_frame
            
            if (skip_pan_frame is not None and 
                diagonal_increase is not None and diagonal_decrease is not None):
                break
        
        direct_candidates = []
        for frame in [prev_pan_frame, prev_tilt_frame]:
            if frame is not None:
                for obj in frame['objects']:
                    direct_candidates.append({
                        **obj,
                        'frame_pan': frame['pan'],
                        'frame_tilt': frame['tilt'],
                        'frame_timestamp': frame['timestamp']
                    })
        
        skip_candidates = []
        for frame in [skip_pan_frame, diagonal_increase, diagonal_decrease]:
            if frame is not None:
                for obj in frame['objects']:
                    skip_candidates.append({
                        **obj,
                        'frame_pan': frame['pan'],
                        'frame_tilt': frame['tilt'],
                        'frame_timestamp': frame['timestamp']
                    })
        
        return {'direct': direct_candidates, 'skip': skip_candidates}
    
    def _hungarian_matching(self, curr_objects, all_candidates):
        """헝가리안 알고리즘 매칭"""
        if not all_candidates:
            track_ids = []
            for obj in curr_objects:
                track_id = self.next_id
                self.next_id += 1
                obj['track_id'] = track_id
                track_ids.append(track_id)
            return track_ids
        
        n_objects = len(curr_objects)
        n_candidates = len(all_candidates)
        
        cost_matrix = np.ones((n_objects, n_candidates))
        similarity_matrix = np.zeros((n_objects, n_candidates))
        
        for i, obj in enumerate(curr_objects):
            for j, candidate in enumerate(all_candidates):
                sim = calc_cosine_similarity(obj['vec'], candidate['vec'])
                similarity_matrix[i, j] = sim
                cost_matrix[i, j] = 1.0 - sim
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        obj_assignments = {}
        for obj_idx, cand_idx in zip(row_ind, col_ind):
            candidate = all_candidates[cand_idx]
            sim = similarity_matrix[obj_idx, cand_idx]
            
            threshold = 0.5
            if sim >= threshold:
                obj_assignments[obj_idx] = (candidate['track_id'], sim)
        
        track_ids = []
        for obj_idx, obj in enumerate(curr_objects):
            if obj_idx in obj_assignments:
                track_id, _ = obj_assignments[obj_idx]
            else:
                track_id = self.next_id
                self.next_id += 1
            
            obj['track_id'] = track_id
            track_ids.append(track_id)
        
        return track_ids


class ObjectTrackerNoGrid(ObjectTrackerOriginal):
    """버전 2: No Grid + 5개 프레임 타입 후보"""
    def __init__(self):
        super().__init__()
        self.version_name = "V2_NoGrid_Spatial"
        
    def add_detections(self, boxes, scores, img_on, diff, pan, tilt, timestamp):
        curr_objects = []
        H, W = img_on.shape[:2]
        
        for i, (x, y, w, h) in enumerate(boxes):
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            
            half_size = ROI_SIZE // 2
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(W, center_x + half_size)
            y2 = min(H, center_y + half_size)
            
            roi = img_on[y1:y2, x1:x2]
            diff_roi = diff[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
                
            # ⭐ No Grid 특징
            vec = get_feature_vector_no_grid(roi, diff_roi=diff_roi)
            
            unique_id = str(self.unique_id_counter)
            self.unique_id_counter += 1
            
            curr_objects.append({
                'box': (x, y, w, h),
                'vec': vec,
                'idx': i,
                'unique_id': unique_id,
                'roi_img': roi.copy()
            })
        
        # 동일한 공간 기반 후보 선택
        candidates_dict = self._find_spatial_candidates(pan, tilt)
        all_candidates = candidates_dict['direct'] + candidates_dict['skip']
        
        track_ids = self._hungarian_matching(curr_objects, all_candidates)
        
        self.frames.append({
            'pan': pan,
            'tilt': tilt,
            'timestamp': timestamp,
            'objects': curr_objects
        })
        
        return track_ids


class ObjectTrackerTemporal(ObjectTrackerBase):
    """버전 3: Grid 11x11 + 시간 순서 최근 5개 프레임"""
    def __init__(self, n_frames=5):
        super().__init__(f"V3_Grid_Temporal_n{n_frames}")
        self.n_frames = n_frames
        
    def add_detections(self, boxes, scores, img_on, diff, pan, tilt, timestamp):
        curr_objects = []
        H, W = img_on.shape[:2]
        
        for i, (x, y, w, h) in enumerate(boxes):
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            
            half_size = ROI_SIZE // 2
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(W, center_x + half_size)
            y2 = min(H, center_y + half_size)
            
            roi = img_on[y1:y2, x1:x2]
            diff_roi = diff[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
                
            vec = get_feature_vector_grid(roi, diff_roi=diff_roi, grid_size=(11, 11))
            
            unique_id = str(self.unique_id_counter)
            self.unique_id_counter += 1
            
            curr_objects.append({
                'box': (x, y, w, h),
                'vec': vec,
                'idx': i,
                'unique_id': unique_id,
                'roi_img': roi.copy()
            })
        
        # ⭐ 시간 순서로 최근 n개 프레임의 모든 객체
        all_candidates = self._find_temporal_candidates()
        
        track_ids = self._hungarian_matching(curr_objects, all_candidates)
        
        self.frames.append({
            'pan': pan,
            'tilt': tilt,
            'timestamp': timestamp,
            'objects': curr_objects
        })
        
        return track_ids
    
    def _find_temporal_candidates(self):
        """시간 순서로 최근 n개 프레임의 모든 객체"""
        candidates = []
        
        # 최근 n개 프레임
        for i in range(min(self.n_frames, len(self.frames))):
            frame = self.frames[-(i+1)]
            for obj in frame['objects']:
                candidates.append({
                    **obj,
                    'frame_pan': frame['pan'],
                    'frame_tilt': frame['tilt'],
                    'frame_timestamp': frame['timestamp']
                })
        
        return candidates
    
    def _hungarian_matching(self, curr_objects, all_candidates):
        """헝가리안 알고리즘 매칭"""
        if not all_candidates:
            track_ids = []
            for obj in curr_objects:
                track_id = self.next_id
                self.next_id += 1
                obj['track_id'] = track_id
                track_ids.append(track_id)
            return track_ids
        
        n_objects = len(curr_objects)
        n_candidates = len(all_candidates)
        
        cost_matrix = np.ones((n_objects, n_candidates))
        similarity_matrix = np.zeros((n_objects, n_candidates))
        
        for i, obj in enumerate(curr_objects):
            for j, candidate in enumerate(all_candidates):
                sim = calc_cosine_similarity(obj['vec'], candidate['vec'])
                similarity_matrix[i, j] = sim
                cost_matrix[i, j] = 1.0 - sim
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        obj_assignments = {}
        for obj_idx, cand_idx in zip(row_ind, col_ind):
            candidate = all_candidates[cand_idx]
            sim = similarity_matrix[obj_idx, cand_idx]
            
            threshold = 0.5
            if sim >= threshold:
                obj_assignments[obj_idx] = (candidate['track_id'], sim)
        
        track_ids = []
        for obj_idx, obj in enumerate(curr_objects):
            if obj_idx in obj_assignments:
                track_id, _ = obj_assignments[obj_idx]
            else:
                track_id = self.next_id
                self.next_id += 1
            
            obj['track_id'] = track_id
            track_ids.append(track_id)
        
        return track_ids


# =========================================================
# 스캔 이미지 파싱
# =========================================================
def parse_scan_images(scan_folder):
    """스캔 폴더에서 이미지 파싱"""
    folder = Path(scan_folder)
    images = []
    
    for img_file in folder.glob("*.jpg"):
        if '.ud' not in img_file.name:
            continue
            
        match = re.search(r't([+-]?\d+)_p([+-]?\d+)_(\d{8}_\d{6}_\d{3})_(led_on|led_off)\.ud', img_file.name)
        if not match:
            continue
        
        tilt = int(match.group(1))
        pan = int(match.group(2))
        timestamp = match.group(3)
        led_type = 'on' if 'led_on' in match.group(4) else 'off'
        
        images.append((pan, tilt, led_type, str(img_file), timestamp))
    
    images.sort(key=lambda x: x[4])
    return images


# =========================================================
# 저장 함수
# =========================================================
def save_tracked_objects(tracker, output_folder):
    """각 track_id별로 ROI 그리드 저장"""
    os.makedirs(output_folder, exist_ok=True)
    
    tracks = {}
    for (pan, tilt), objects in tracker.frame_objects.items():
        for obj in objects:
            track_id = obj['track_id']
            if track_id not in tracks:
                tracks[track_id] = []
            tracks[track_id].append({
                'pan': pan,
                'tilt': tilt,
                'box': obj['box'],
                'roi_img': obj.get('roi_img', None),
                'unique_id': obj.get('unique_id', 'N/A')
            })
    
    for track_id, detections in tracks.items():
        valid_rois = [d for d in detections if d['roi_img'] is not None]
        if not valid_rois:
            continue
        
        num_imgs = len(valid_rois)
        cols = min(10, num_imgs)
        rows = (num_imgs + cols - 1) // cols
        
        STANDARD_SIZE = (150, 150)
        roi_w, roi_h = STANDARD_SIZE
        
        grid_h = rows * (roi_h + 10) + 10
        grid_w = cols * (roi_w + 10) + 10
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        for idx, det in enumerate(valid_rois):
            row = idx // cols
            col = idx % cols
            
            y_start = row * (roi_h + 10) + 10
            x_start = col * (roi_w + 10) + 10
            
            roi = det['roi_img']
            roi_resized = cv2.resize(roi, STANDARD_SIZE)
            
            canvas[y_start:y_start+roi_h, x_start:x_start+roi_w] = roi_resized
            
            unique_id = det.get('unique_id', 'N/A')
            pan_tilt = f"P{det['pan']:+d}T{det['tilt']:+d}"
            
            cv2.rectangle(canvas, (x_start, y_start), (x_start+STANDARD_SIZE[0], y_start+15), (0, 0, 0), -1)
            cv2.putText(canvas, unique_id, (x_start+2, y_start+12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            
            cv2.rectangle(canvas, (x_start, y_start+15), (x_start+80, y_start+30), (0, 0, 0), -1)
            cv2.putText(canvas, pan_tilt, (x_start+2, y_start+27),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        output_path = os.path.join(output_folder, f"track_id_{track_id:03d}.jpg")
        cv2.imwrite(output_path, canvas)


# =========================================================
# 메인 실행
# =========================================================
def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ 모델 파일 없음")
        return
    
    model = YOLO(MODEL_PATH)
    
    # 3가지 Tracker 생성
    trackers = {
        'v1_grid_spatial': ObjectTrackerOriginal(),
        'v2_nogrid_spatial': ObjectTrackerNoGrid(),
        'v3_grid_temporal': ObjectTrackerTemporal(n_frames=5)
    }
    
    print("=" * 80)
    print("🎯 MOT 알고리즘 비교 실험 시작 (3가지 버전)")
    print("=" * 80)
    print(f"버전 1: {trackers['v1_grid_spatial'].version_name}")
    print(f"버전 2: {trackers['v2_nogrid_spatial'].version_name}")
    print(f"버전 3: {trackers['v3_grid_temporal'].version_name}\n")
    
    # 스캔 이미지 파싱
    images = parse_scan_images(SCAN_FOLDER)
    
    # ON/OFF 페어 생성
    pairs = {}
    for pan, tilt, led_type, filepath, timestamp in images:
        key = (pan, tilt)
        if key not in pairs:
            pairs[key] = {}
        pairs[key][led_type] = {'path': filepath, 'timestamp': timestamp}
    
    complete_pairs = [k for k, v in pairs.items() if 'on' in v and 'off' in v]
    sorted_keys = sorted(complete_pairs, key=lambda x: pairs[x]['on']['timestamp'])
    
    print(f"✅ {len(sorted_keys)}개 ON/OFF 페어 발견\n")
    
    # 모든 Tracker에 동일한 데이터 처리
    for idx, (pan, tilt) in enumerate(sorted_keys):
        print(f"\r진행중: {idx+1}/{len(sorted_keys)} (P{pan:+d} T{tilt:+d})", end='')
        
        pair = pairs[(pan, tilt)]
        timestamp = pair['on']['timestamp']
        
        img_on = cv2.imread(pair['on']['path'])
        img_off = cv2.imread(pair['off']['path'])
        
        if img_on is None or img_off is None:
            continue
        
        diff = cv2.absdiff(img_on, img_off)
        
        boxes, scores, classes = predict_with_tiling(
            model, diff, rows=2, cols=3, overlap=0.15,
            conf=CONF_THRES, iou=IOU_THRES
        )
        
        if len(boxes) == 0:
            continue
        
        # 3가지 Tracker에 모두 적용
        for tracker_name, tracker in trackers.items():
            track_ids = tracker.add_detections(boxes, scores, img_on, diff, pan, tilt, timestamp)
        
        # 진행 상황 출력 (hungarian_final 스타일)
        print(f"[Pan={pan:+4d}, Tilt={tilt:+3d}] {len(boxes)}개 검출")
    
    # ⭐ Track 병합 (hungarian_final과 동일)
    print("\n" + "="*80)
    print("✅ 추적 완료! Track 병합 중...")
    print("="*80 + "\n")
    
    for tracker_name, tracker in trackers.items():
        print(f"\n[{tracker.version_name}] Track 병합")
        tracker.merge_similar_tracks(merge_threshold=0.4, min_detections=3)
    
    # 각 Tracker별 요약 출력
    print("\n" + "="*80)
    print("✅ 병합 완료! 최종 통계")
    print("="*80)
    
    for tracker_name, tracker in trackers.items():
        tracks = {}
        for frame in tracker.frames:
            for obj in frame['objects']:
                tid = obj['track_id']
                if tid not in tracks:
                    tracks[tid] = 0
                tracks[tid] += 1
        
        total_tracks = len(tracks)
        total_detections = sum(tracks.values())
        
        print(f"\n[{tracker.version_name}]")
        print(f"  총 검출: {total_detections}개")
        print(f"  부여된 고유 ID: 0 ~ {tracker.next_id - 1} ({tracker.next_id}개)")
        print(f"  최종 Track 수: {total_tracks}개")
    
    print("\n" + "="*80)
    print("결과 저장 중...")
    print("="*80 + "\n")
    
    # 결과 저장
    results = {}
    for tracker_name, tracker in trackers.items():
        scheme_folder = f"./mot_comparison_results/{tracker_name}"
        os.makedirs(scheme_folder, exist_ok=True)
        
        # ROI 이미지 저장
        save_tracked_objects(tracker, scheme_folder)
        
        # 통계 계산
        tracks = {}
        for frame in tracker.frames:
            for obj in frame['objects']:
                tid = obj['track_id']
                if tid not in tracks:
                    tracks[tid] = 0
                tracks[tid] += 1
        
        total_tracks = len(tracks)
        total_detections = sum(tracks.values())
        avg_detections = total_detections / total_tracks if total_tracks > 0 else 0
        
        results[tracker_name] = {
            'version': tracker.version_name,
            'total_tracks': total_tracks,
            'total_detections': total_detections,
            'avg_detections_per_track': round(avg_detections, 2),
            'track_distribution': tracks
        }
        
        # 요약 저장
        summary_path = os.path.join(scheme_folder, "summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"MOT 결과 요약 - {tracker.version_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"총 Track 수: {total_tracks}개\n")
            f.write(f"총 검출 수: {total_detections}개\n")
            f.write(f"Track당 평균 검출: {avg_detections:.2f}개\n\n")
            f.write(f"Track 분포:\n")
            for tid, count in sorted(tracks.items()):
                f.write(f"  Track {tid:3d}: {count:3d}개\n")
        
        print(f"💾 {tracker.version_name}: Track ID별 이미지 저장 중...")
        print(f"✅ 저장 완료! → {scheme_folder}/ 폴더 확인")
        print(f"   총{total_tracks}개 Track, {total_detections}개 검출\n")
    
    # 전체 비교 JSON 저장
    comparison_file = "./mot_comparison_summary.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 전체 비교 결과 저장: {comparison_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
