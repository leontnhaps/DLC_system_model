"""
MOT Scan Simulation Test
기존 스캔 이미지 폴더로 전체 추적 알고리즘 테스트
"""
import cv2
import numpy as np
import sys
import os
import re
from pathlib import Path
from numpy.linalg import norm
from ultralytics import YOLO

# ---------------------------------------------------------
# 기존 모듈 로드
# ---------------------------------------------------------
# ⭐ Detection_test 폴더를 sys.path에 추가
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

# ⭐ 스캔 폴더 베이스 경로
BASE_FOLDER = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\추계 이후자료\Diff YOLO Dataset"
# 처리할 폴더 목록
# FOLDER_NAMES = ["젤먼거", "젤먼거2", "젤먼거3", "젤먼거4", "젤먼거5", 
#                 "젤먼거6", "젤먼거7", "젤먼거8", "젤먼거9", "젤먼거10"]
FOLDER_NAMES = ["젤먼거"]

CONF_THRES = 0.50
IOU_THRES = 0.2
# ⭐ 고정 ROI 크기 (중심 기준)
ROI_SIZE = 300  # 200x200 픽셀

# =========================================================
# 특징 추출 (HSV + Grayscale 결합)
# =========================================================
def get_feature_vector(roi_bgr, diff_roi=None, grid_size=(11, 11)):
    """
    격자 기반 히스토그램 추출: 공간적 위치 정보를 포함함
    ⭐ HSV + Grayscale 히스토그램 결합 (Diff 마스크 적용)
    grid_size: (rows, cols) - ROI를 나눌 구역 수
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = hsv.shape[:2]
    rows, cols = grid_size
    
    feature_vector = []
    
    # ROI를 격자로 나누어 각 구역의 히스토그램 계산
    for r in range(rows):
        for c in range(cols):
            # 구역 좌표 계산
            y_start = int((r / rows) * h)
            y_end = int(((r + 1) / rows) * h)
            x_start = int((c / cols) * w)
            x_end = int(((c + 1) / cols) * w)
            
            # 구역(Cell) 추출
            cell_hsv = hsv[y_start:y_end, x_start:x_end]
            cell_gray = gray[y_start:y_end, x_start:x_end]
            
            # ⭐ Diff 기반 마스크 생성
            if diff_roi is not None:
                # Diff cell 추출
                diff_cell = diff_roi[y_start:y_end, x_start:x_end]
                # Grayscale로 변환
                if len(diff_cell.shape) == 3:
                    diff_gray = cv2.cvtColor(diff_cell, cv2.COLOR_BGR2GRAY)
                else:
                    diff_gray = diff_cell
                
                # ⭐ Diff < 20인 부분만 (배경 부분, 객체 필름 제외!)
                # Diff가 작은 부분 = 변화 없는 배경 → 사용
                # Diff가 큰 부분 = LED 변화 객체(필름) → 제외
                diff_mask = (diff_gray < 20).astype(np.uint8) * 255
                
                # V > 30 조건과 결합
                v_mask = cv2.inRange(cell_hsv, (0, 0, 30), (180, 255, 255))
                mask = cv2.bitwise_and(diff_mask, v_mask)
            else:
                # Diff가 없으면 기본 마스크만
                mask = cv2.inRange(cell_hsv, (0, 0, 30), (180, 255, 255))
            
            # ⭐ 1. HSV 히스토그램 (Hue + Saturation)
            # Hue: 8 bins, Saturation: 4 bins → 32차원
            hist_hsv = cv2.calcHist([cell_hsv], [0, 1], mask, [8, 4], [0, 180, 0, 256])
            cv2.normalize(hist_hsv, hist_hsv, 0, 1, cv2.NORM_MINMAX)
            
            # ⭐ 2. Grayscale 히스토그램
            # 16 bins → 16차원
            hist_gray = cv2.calcHist([cell_gray], [0], mask, [16], [0, 256])
            cv2.normalize(hist_gray, hist_gray, 0, 1, cv2.NORM_MINMAX)
            
            # ⭐ 3. 두 히스토그램 결합 (32 + 16 = 48차원)
            combined_hist = np.concatenate([hist_hsv.flatten(), hist_gray.flatten()])
            feature_vector.append(combined_hist)
    
    # 모든 구역의 히스토그램을 하나로 결합 (공간 정보가 순서대로 쌓임)
    final_vector = np.concatenate(feature_vector)
    
    # 최종 벡터 정규화 (코사인 유사도 계산 최적화)
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

def save_tracked_objects(tracker, output_folder="./mot_output"):
    """
    각 track_id별로 검출된 모든 ROI를 그리드로 저장
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # track_id별로 ROI 수집
    tracks = {}  # {track_id: [(img, pan, tilt), ...]}
    
    for (pan, tilt), objects in tracker.frame_objects.items():
        for obj in objects:
            track_id = obj['track_id']
            if track_id not in tracks:
                tracks[track_id] = []
            # ⭐ unique_id 포함하여 저장
            tracks[track_id].append({
                'pan': pan,
                'tilt': tilt,
                'box': obj['box'],
                'roi_img': obj.get('roi_img', None),
                'unique_id': obj.get('unique_id', 'N/A')  # ⭐ 고유 ID 추가
            })
    
    # 각 track_id별 이미지 생성
    for track_id, detections in tracks.items():
        if not detections:
            continue
        
        # 유효한 ROI만 필터링
        valid_rois = [d for d in detections if d['roi_img'] is not None]
        if not valid_rois:
            continue
        
        num_imgs = len(valid_rois)
        cols = min(10, num_imgs)  # 최대 10열
        rows = (num_imgs + cols - 1) // cols
        
        # ⭐ 표준 ROI 크기 설정 (모든 ROI를 같은 크기로 리사이즈)
        STANDARD_SIZE = (150, 150)  # (width, height)
        roi_w, roi_h = STANDARD_SIZE
        
        # 그리드 캔버스 생성
        grid_h = rows * (roi_h + 10) + 10
        grid_w = cols * (roi_w + 10) + 10
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        # ROI 배치
        for idx, det in enumerate(valid_rois):
            row = idx // cols
            col = idx % cols
            
            y_start = row * (roi_h + 10) + 10
            x_start = col * (roi_w + 10) + 10
            
            # ⭐ ROI를 표준 크기로 리사이즈
            roi = det['roi_img']
            roi_resized = cv2.resize(roi, STANDARD_SIZE)
            
            canvas[y_start:y_start+roi_h, x_start:x_start+roi_w] = roi_resized
            
            # ⭐ 고유 ID와 Pan/Tilt 정보 표시
            unique_id = det.get('unique_id', 'N/A')
            pan_tilt = f"P{det['pan']:+d}T{det['tilt']:+d}"
            
            # 첫 줄: 고유 ID (검은 배경)
            cv2.rectangle(canvas, (x_start, y_start), (x_start+STANDARD_SIZE[0], y_start+15), (0, 0, 0), -1)
            cv2.putText(canvas, unique_id, (x_start+2, y_start+12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            
            # 둘째 줄: Pan/Tilt (검은 배경)
            cv2.rectangle(canvas, (x_start, y_start+15), (x_start+80, y_start+30), (0, 0, 0), -1)
            cv2.putText(canvas, pan_tilt, (x_start+2, y_start+27),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # 저장
        output_path = os.path.join(output_folder, f"track_id_{track_id:03d}.jpg")
        cv2.imwrite(output_path, canvas)
        print(f"  💾 Track ID {track_id}: {num_imgs}개 저장 → {output_path}")

# =========================================================
# MOT Tracker (타임스탬프 기반 순차 추적)
# =========================================================
class ObjectTracker:
    def __init__(self):
        self.next_id = 0
        # 순차적으로 프레임 저장
        self.frames = []  # [(pan, tilt, timestamp, objects), ...]
        # ⭐ 유사도 로그
        self.similarity_log = []  # 모든 비교 기록
        # ⭐ 고유 ID 카운터 (1부터 시작)
        self.unique_id_counter = 1
        
    def reset(self):
        self.next_id = 0
        self.frames = []
        self.similarity_log = []
        self.unique_id_counter = 1
        
    def add_detections(self, boxes, scores, img_on, diff, pan, tilt, timestamp):
        """
        타임스탬프 기반 순차 추적:
        1. 직전 프레임 (threshold=0.3)
        2. 프레임 건너뛰기 (threshold=0.4) - 검출 놓침 대비
        ⭐ diff 이미지를 받아서 필름 배경 필터링
        """
        # 현재 프레임 특징 추출
        curr_objects = []
        H, W = img_on.shape[:2]
        
        for i, (x, y, w, h) in enumerate(boxes):
            # ⭐ 객체 중심 계산
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            
            # ⭐ 중심 기준 고정 크기 ROI
            half_size = ROI_SIZE // 2
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(W, center_x + half_size)
            y2 = min(H, center_y + half_size)
            
            roi = img_on[y1:y2, x1:x2]
            diff_roi = diff[y1:y2, x1:x2]  # ⭐ Diff ROI도 추출
            
            if roi.size == 0:
                continue
                
            # ⭐ diff_roi 전달하여 필름 필터링
            vec = get_feature_vector(roi, diff_roi=diff_roi)
            
            # ⭐ 고유 ID 생성
            curr_objects.append({
                'box': (x, y, w, h),
                'vec': vec,
                'idx': i,
                'roi_img': roi.copy(),
                'unique_id': None
            })
        
        # 이전 프레임들에서 후보 찾기 (⭐ 딕셔너리 반환)
        candidates_dict = self._find_prev_candidates(pan, tilt)
        direct_candidates = candidates_dict['direct']  # n-1 프레임
        skip_candidates = candidates_dict['skip']       # n-2 프레임
        
        # ⭐ 2단계 글로벌 매칭 알고리즘
        track_ids = []
        
        # 1. 고유 ID 먼저 할당
        for obj_idx, obj in enumerate(curr_objects):
            unique_id = str(self.unique_id_counter)
            self.unique_id_counter += 1
            obj['unique_id'] = unique_id
        
        # 2-1. 직접 후보 매칭 (threshold = 0.3)
        direct_matches = []
        for obj_idx, obj in enumerate(curr_objects):
            for candidate in direct_candidates:
                sim = calc_cosine_similarity(obj['vec'], candidate['vec'])
                direct_matches.append((sim, obj_idx, candidate, 'direct'))
        
        # 2-2. 건너뛰기 후보 매칭 (threshold = 0.4)
        skip_matches = []
        for obj_idx, obj in enumerate(curr_objects):
            for candidate in skip_candidates:
                sim = calc_cosine_similarity(obj['vec'], candidate['vec'])
                skip_matches.append((sim, obj_idx, candidate, 'skip'))
        
        # 3. 직접 후보를 우선 정렬 (유사도 높은 순)
        direct_matches.sort(key=lambda x: x[0], reverse=True)
        skip_matches.sort(key=lambda x: x[0], reverse=True)
        
        # 4. 탐욕적 할당 (2단계)
        used_objects = set()
        used_track_ids = set()
        obj_assignments = {}  # {obj_idx: (track_id, similarity, candidate, source)}
        
        # 4-1. 먼저 직접 후보로 매칭 시도 (threshold = 0.3)
        for sim, obj_idx, candidate, source in direct_matches:
            if obj_idx in used_objects or candidate['track_id'] in used_track_ids:
                continue
            
            if sim > 0.3:  # 직접 후보 threshold
                obj_assignments[obj_idx] = (candidate['track_id'], sim, candidate, source)
                used_objects.add(obj_idx)
                used_track_ids.add(candidate['track_id'])
        
        # 4-2. 매칭 실패한 객체는 건너뛰기 후보로 시도 (threshold = 0.4)
        for sim, obj_idx, candidate, source in skip_matches:
            if obj_idx in used_objects or candidate['track_id'] in used_track_ids:
                continue
            
            if sim > 0.35:  # 건너뛰기 후보 threshold (더 엄격)
                obj_assignments[obj_idx] = (candidate['track_id'], sim, candidate, source)
                used_objects.add(obj_idx)
                used_track_ids.add(candidate['track_id'])
        
        # 5. 최종 track_id 할당 및 로그 생성
        # 모든 후보 합치기 (로그용)
        all_candidates = direct_candidates + skip_candidates
        
        for obj_idx, obj in enumerate(curr_objects):
            if obj_idx in obj_assignments:
                # 매칭 성공
                track_id, best_sim, best_candidate, source = obj_assignments[obj_idx]
            else:
                # 매칭 실패 → 새 ID
                track_id = self.next_id
                self.next_id += 1
                best_sim = 0.0
                best_candidate = None
                source = None
            
            obj['track_id'] = track_id
            track_ids.append(track_id)
            
            # ⭐ 로그 생성 (모든 후보와의 비교 기록)
            log_entry = {
                'pan': pan,
                'tilt': tilt,
                'timestamp': timestamp,
                'obj_idx': obj_idx,
                'unique_id': obj['unique_id'],
                'comparisons': []
            }
            
            for candidate in all_candidates:
                sim = calc_cosine_similarity(obj['vec'], candidate['vec'])
                log_entry['comparisons'].append({
                    'candidate_id': candidate['track_id'],
                    'candidate_unique_id': candidate.get('unique_id', 'N/A'),
                    'candidate_pan': candidate['frame_pan'],
                    'candidate_tilt': candidate['frame_tilt'],
                    'candidate_timestamp': candidate['frame_timestamp'],
                    'similarity': float(sim)
                })
            
            log_entry['assigned_id'] = track_id
            log_entry['best_similarity'] = float(best_sim)
            log_entry['is_new_object'] = (best_candidate is None)
            log_entry['match_source'] = source  # 'direct', 'skip', or None
            
            self.similarity_log.append(log_entry)
        
        # 현재 프레임 저장
        self.frames.append({
            'pan': pan,
            'tilt': tilt,
            'timestamp': timestamp,
            'objects': curr_objects
        })
        
        return track_ids
    
    def _find_prev_candidates(self, current_pan, current_tilt):
        """
        프레임 후보 검색:
        1. n-1 (최근 1프레임): 같은 Pan, 같은 Tilt
        2. n-1 대각선 1: (Pan 변화 AND Tilt 변화) - fallback용
        3. n-2 (2프레임 전): 같은 Pan, 같은 Tilt
        4. ⭐ 양방향 대각선: 지그재그 스캔 대응 (threshold 0.4)
        
        반환: {'direct': [...], 'skip': [...]}
        """
        if not self.frames:
            return {'direct': [], 'skip': []}
        
        # n-1 프레임 (직접 이웃)
        prev_pan_frame = None
        prev_tilt_frame = None
        prev_diagonal_frame = None  # 기본 대각선 (fallback용)
        
        # n-2 프레임 (프레임 건너뛰기)
        skip_pan_frame = None
        skip_tilt_frame = None
        
        # ⭐ 양방향 대각선 (지그재그 스캔 대응)
        diagonal_increase = []  # Pan 증가, Tilt 변화
        diagonal_decrease = []  # Pan 감소, Tilt 변화
        
        # 최근 프레임부터 역순 탐색
        for i in range(len(self.frames)):
            prev_frame = self.frames[-(i+1)]
            frame_pan = prev_frame['pan']
            frame_tilt = prev_frame['tilt']
            
            # ⭐ n-1 프레임 검색
            if i == 0:  # 가장 최근 프레임
                # 같은 Pan, 다른 Tilt
                if frame_pan == current_pan and frame_tilt != current_tilt and prev_pan_frame is None:
                    prev_pan_frame = prev_frame
                
                # 같은 Tilt, 다른 Pan
                if frame_tilt == current_tilt and frame_pan != current_pan and prev_tilt_frame is None:
                    prev_tilt_frame = prev_frame
                
                # 기본 대각선 (fallback용, 다른 Pan AND 다른 Tilt)
                if frame_pan != current_pan and frame_tilt != current_tilt and prev_diagonal_frame is None:
                    prev_diagonal_frame = prev_frame
            
            # ⭐ n-2 프레임 검색 (프레임 건너뛰기)
            elif i == 1:  # 2프레임 전
                # 같은 Pan, 다른 Tilt
                if frame_pan == current_pan and frame_tilt != current_tilt and skip_pan_frame is None:
                    skip_pan_frame = prev_frame
                
                # 같은 Tilt, 다른 Pan
                if frame_tilt == current_tilt and frame_pan != current_pan and skip_tilt_frame is None:
                    skip_tilt_frame = prev_frame
            
            # ⭐ 양방향 대각선 검색 (소수만, 최대 2개씩)
            if frame_pan != current_pan and frame_tilt != current_tilt:
                # Pan 증가 방향 (→)
                if frame_pan < current_pan and len(diagonal_increase) < 2:
                    diagonal_increase.append(prev_frame)
                
                # Pan 감소 방향 (←)
                if frame_pan > current_pan and len(diagonal_decrease) < 2:
                    diagonal_decrease.append(prev_frame)
            
            # 충분히 수집했으면 종료
            if (skip_pan_frame is not None and skip_tilt_frame is not None and 
                len(diagonal_increase) >= 2 and len(diagonal_decrease) >= 2):
                break
        
        # ⭐ n-1 후보 수집 (direct)
        direct_candidates = []
        
        # Pan 방향
        if prev_pan_frame is not None:
            for obj in prev_pan_frame['objects']:
                direct_candidates.append({
                    **obj,
                    'frame_pan': prev_pan_frame['pan'],
                    'frame_tilt': prev_pan_frame['tilt'],
                    'frame_timestamp': prev_pan_frame['timestamp']
                })
        
        # Tilt 방향
        if prev_tilt_frame is not None:
            for obj in prev_tilt_frame['objects']:
                direct_candidates.append({
                    **obj,
                    'frame_pan': prev_tilt_frame['pan'],
                    'frame_tilt': prev_tilt_frame['tilt'],
                    'frame_timestamp': prev_tilt_frame['timestamp']
                })
        
        # 기본 대각선 fallback (둘 다 없을 때만)
        if prev_pan_frame is None and prev_tilt_frame is None and prev_diagonal_frame is not None:
            for obj in prev_diagonal_frame['objects']:
                direct_candidates.append({
                    **obj,
                    'frame_pan': prev_diagonal_frame['pan'],
                    'frame_tilt': prev_diagonal_frame['tilt'],
                    'frame_timestamp': prev_diagonal_frame['timestamp']
                })
        
        # ⭐ skip 후보 수집 (n-2 + 양방향 대각선)
        skip_candidates = []
        
        # n-2 Pan 방향
        if skip_pan_frame is not None:
            for obj in skip_pan_frame['objects']:
                skip_candidates.append({
                    **obj,
                    'frame_pan': skip_pan_frame['pan'],
                    'frame_tilt': skip_pan_frame['tilt'],
                    'frame_timestamp': skip_pan_frame['timestamp']
                })
        
        # n-2 Tilt 방향
        if skip_tilt_frame is not None:
            for obj in skip_tilt_frame['objects']:
                skip_candidates.append({
                    **obj,
                    'frame_pan': skip_tilt_frame['pan'],
                    'frame_tilt': skip_tilt_frame['tilt'],
                    'frame_timestamp': skip_tilt_frame['timestamp']
                })
        
        # ⭐ 양방향 대각선 (지그재그 스캔 대응)
        for diag_frame in diagonal_increase + diagonal_decrease:
            for obj in diag_frame['objects']:
                skip_candidates.append({
                    **obj,
                    'frame_pan': diag_frame['pan'],
                    'frame_tilt': diag_frame['tilt'],
                    'frame_timestamp': diag_frame['timestamp']
                })
        
        return {'direct': direct_candidates, 'skip': skip_candidates}
    
    def _match_object_with_log(self, curr_obj, candidates, pan, tilt, timestamp, obj_idx, unique_id):
        """후보들과 비교하여 최적 매칭 (⭐ 로그 포함)"""
        best_match_id = None
        best_sim = 0.3  # 임계값
        
        # ⭐ 로그 엔트리
        log_entry = {
            'pan': pan,
            'tilt': tilt,
            'timestamp': timestamp,
            'obj_idx': obj_idx,
            'unique_id': unique_id,  # ⭐ 고유 ID 추가
            'comparisons': []
        }
        
        for candidate in candidates:
            sim = calc_cosine_similarity(curr_obj['vec'], candidate['vec'])
            
            # ⭐ 각 비교 기록
            log_entry['comparisons'].append({
                'candidate_id': candidate['track_id'],
                'candidate_unique_id': candidate.get('unique_id', 'N/A'),  # ⭐ 후보 고유 ID
                'candidate_pan': candidate['frame_pan'],
                'candidate_tilt': candidate['frame_tilt'],
                'candidate_timestamp': candidate['frame_timestamp'],
                'similarity': float(sim)
            })
            
            if sim > best_sim:
                best_sim = sim
                best_match_id = candidate['track_id']
        
        # 매칭 실패 시 새 ID 부여
        if best_match_id is None:
            best_match_id = self.next_id
            self.next_id += 1
        
        # ⭐ 로그에 최종 결과 추가
        log_entry['assigned_id'] = best_match_id
        log_entry['best_similarity'] = float(best_sim)
        log_entry['is_new_object'] = (best_sim <= 0.5)
        
        return best_match_id, log_entry
    
    @property
    def frame_objects(self):
        """시각화를 위한 호환성 속성"""
        result = {}
        for frame in self.frames:
            key = (frame['pan'], frame['tilt'])
            result[key] = frame['objects']
        return result
    
    def save_similarity_log(self, output_path="./mot_output/similarity_log.txt"):
        """유사도 로그를 텍스트 파일로 저장"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MOT Similarity Log\n")
            f.write("=" * 80 + "\n\n")
            
            for entry in self.similarity_log:
                f.write(f"\n[Frame] Pan={entry['pan']:+4d}, Tilt={entry['tilt']:+3d}, "
                       f"Timestamp={entry['timestamp']}, Object #{entry['obj_idx']}\n")
                f.write(f"  🆔 Unique ID: {entry['unique_id']}\n")  # ⭐ 고유 ID 출력
                f.write(f"  ✅ Assigned Track ID: {entry['assigned_id']} ")
                if entry['is_new_object']:
                    f.write("(NEW OBJECT)\n")
                else:
                    f.write(f"(Best Similarity: {entry['best_similarity']:.4f})\n")
                
                if entry['comparisons']:
                    f.write(f"  🔍 Compared with {len(entry['comparisons'])} candidates:\n")
                    # 유사도 높은 순으로 정렬
                    sorted_comps = sorted(entry['comparisons'], 
                                         key=lambda x: x['similarity'], reverse=True)
                    for comp in sorted_comps:
                        marker = "  ⭐" if comp['candidate_id'] == entry['assigned_id'] else "    "
                        f.write(f"{marker} Track ID {comp['candidate_id']:3d} "
                               f"[{comp.get('candidate_unique_id', 'N/A')}] "
                               f"(Pan={comp['candidate_pan']:+4d}, Tilt={comp['candidate_tilt']:+3d}) "
                               f"→ Sim: {comp['similarity']:.4f}\n")
                else:
                    f.write(f"  ℹ️  No candidates (first detection)\n")
                
                f.write("-" * 80 + "\n")

# =========================================================
# 스캔 이미지 파싱 및 정렬
# =========================================================
def parse_scan_images(scan_folder):
    """
    스캔 폴더에서 이미지 파싱 (_ud 파일만)
    Returns: [(pan, tilt, 'on'/'off', filepath, timestamp), ...]
    """
    folder = Path(scan_folder)
    images = []
    
    for img_file in folder.glob("*.jpg"):
        # ⭐ _ud (undistorted) 파일만 처리
        if '_ud' not in img_file.name:
            continue
            
        # 파일명 파싱: img_t+00_p+000_20251128_221105_941_led_on_ud.jpg
        # 패턴: t[tilt]_p[pan]_[timestamp]_led_[on/off]_ud.jpg
        match = re.search(r't([+-]?\d+)_p([+-]?\d+)_(\d{8}_\d{6}_\d{3})_(led_on|led_off)_ud', img_file.name)
        if not match:
            continue
        
        tilt = int(match.group(1))
        pan = int(match.group(2))
        timestamp = match.group(3)  # '20251128_221105_941'
        led_type = 'on' if 'led_on' in match.group(4) else 'off'
        
        images.append((pan, tilt, led_type, str(img_file), timestamp))
    
    # ⭐ 타임스탬프 기준 정렬 (실제 촬영 순서)
    images.sort(key=lambda x: x[4])
    return images

# =========================================================
# 메인 실행
# =========================================================
def process_folder(scan_folder, output_suffix=""):
    """단일 폴더 처리"""
    model = YOLO(MODEL_PATH)
    tracker = ObjectTracker()
    tracker.reset()
    
    # 스캔 이미지 로드
    print(f"\n📂 스캔 폴더: {scan_folder}")
    images = parse_scan_images(scan_folder)
    print(f"✅ 총 {len(images)}개 이미지 발견")
    
    if not images:
        print("⚠️ 파싱된 이미지가 없습니다!")
        return
    
    # ON/OFF 쌍 만들기
    pairs = {}
    for pan, tilt, led_type, filepath, timestamp in images:
        key = (pan, tilt)
        if key not in pairs:
            pairs[key] = {}
        pairs[key][led_type] = {'path': filepath, 'timestamp': timestamp}
    
    print(f"🔍 ON/OFF 쌍: {len(pairs)}개")
    complete_pairs = [k for k, v in pairs.items() if 'on' in v and 'off' in v]
    print(f"   완전한 쌍 (ON+OFF): {len(complete_pairs)}개")
    
    if not complete_pairs:
        print("⚠️ ON/OFF 쌍이 하나도 없습니다!")
        return
    
    # ⭐ ON 이미지의 타임스탬프 기준으로 정렬
    sorted_keys = sorted(complete_pairs, key=lambda x: pairs[x]['on']['timestamp'])
    
    print("="*60)
    print("🚀 추적 시작!")
    print("="*60)
    
    total_detections = 0
    
    for pan, tilt in sorted_keys:
        pair = pairs[(pan, tilt)]
        timestamp = pair['on']['timestamp']
        
        # 이미지 로드
        img_on = cv2.imread(pair['on']['path'])
        img_off = cv2.imread(pair['off']['path'])
        
        if img_on is None or img_off is None:
            continue
        
        # Diff 계산
        diff = cv2.absdiff(img_on, img_off)
        
        # YOLO 검출
        boxes, scores, classes = predict_with_tiling(
            model, diff, rows=2, cols=3, overlap=0.15,
            conf=CONF_THRES, iou=IOU_THRES
        )
        
        if not boxes:
            print(f"[Pan={pan:+4d}, Tilt={tilt:+3d}] 검출 없음")
            continue
        
        # ⭐ 추적
        track_ids = tracker.add_detections(boxes, scores, img_on, diff, pan, tilt, timestamp)
        
        # 결과 출력
        print(f"[Pan={pan:+4d}, Tilt={tilt:+3d}] {len(boxes)}개 검출 → track_ids: {track_ids}")
        total_detections += len(boxes)
    
    print("\n" + "="*60)
    print("✅ 추적 완료!")
    print(f"총 검출: {total_detections}개")
    print(f"부여된 고유 ID: 0 ~ {tracker.next_id - 1} ({tracker.next_id}개)")
    print("="*60)
    
    # ⭐ 출력 폴더 설정
    output_folder = f"./mot_output{output_suffix}"
    
    # 유사도 로그 저장
    print(f"\n💾 유사도 로그 저장 중... → {output_folder}")
    tracker.save_similarity_log(f"{output_folder}/similarity_log.txt")
    print(f"✅ 유사도 로그 저장 완료!")
    
    # 시각화 저장
    print(f"💾 Track ID별 이미지 저장 중...")
    save_tracked_objects(tracker, output_folder=output_folder)
    print(f"✅ 저장 완료! → {output_folder}/ 폴더 확인")


def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ 모델 파일 없음")
        return
    
    print("="*60)
    print(f"🎯 총 {len(FOLDER_NAMES)}개 폴더 처리 시작")
    print("="*60)
    
    for idx, folder_name in enumerate(FOLDER_NAMES, 1):
        scan_folder = os.path.join(BASE_FOLDER, folder_name)
        
        # 폴더 존재 확인
        if not os.path.exists(scan_folder):
            print(f"\n⚠️ [{idx}/{len(FOLDER_NAMES)}] {folder_name}: 폴더 없음, 건너뜀")
            continue
        
        print(f"\n{'='*60}")
        print(f"📁 [{idx}/{len(FOLDER_NAMES)}] {folder_name} 처리 중...")
        print(f"{'='*60}")
        
        # 출력 폴더 suffix (젤먼거 → "", 젤먼거2 → "_2", ...)
        if folder_name == "젤먼거":
            output_suffix = ""
        else:
            # "젤먼거2" → "_2"
            suffix_num = folder_name.replace("젤먼거", "")
            output_suffix = f"_{suffix_num}" if suffix_num else ""
        
        try:
            process_folder(scan_folder, output_suffix)
        except Exception as e:
            print(f"\n❌ {folder_name} 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("🎉 전체 폴더 처리 완료!")
    print("="*60)

if __name__ == "__main__":
    main()

