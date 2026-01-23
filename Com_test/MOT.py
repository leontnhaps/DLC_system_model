#!/usr/bin/env python3
"""
Multi-Object Tracking (MOT) Module
타임스탬프 기반 순차 추적 알고리즘
HSV + Grayscale 히스토그램 특징 추출 및 코사인 유사도 기반 매칭
"""

import cv2
import numpy as np
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment  # ⭐ 헝가리안 알고리즘


# =========================================================
# 특징 추출 (HSV + Grayscale 결합)
# =========================================================
def get_feature_vector(roi_bgr, diff_roi=None, grid_size=(11, 11)):
    """
    격자 기반 히스토그램 추출: 공간적 위치 정보를 포함함
    ⭐ HSV + Grayscale 히스토그램 결합 (Diff 마스크 적용)
    
    Args:
        roi_bgr: BGR 이미지 ROI
        diff_roi: Diff 이미지 ROI (배경 필터링용)
        grid_size: (rows, cols) - ROI를 나눌 구역 수
    
    Returns:
        정규화된 특징 벡터 (numpy array)
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
    """코사인 유사도 계산"""
    if vec_a is None or vec_b is None:
        return 0.0
    dot = np.dot(vec_a, vec_b)
    n_a, n_b = norm(vec_a), norm(vec_b)
    if n_a == 0 or n_b == 0:
        return 0.0
    return dot / (n_a * n_b)


# =========================================================
# MOT Tracker (타임스탬프 기반 순차 추적)
# =========================================================
class ObjectTracker:
    """
    Multi-Object Tracker
    타임스탬프 기반 순차 추적 알고리즘
    2단계 글로벌 매칭 (직전 프레임 + 프레임 건너뛰기)
    """
    
    def __init__(self, roi_size=300, grid_size=(11, 11)):
        """
        Args:
            roi_size: 고정 ROI 크기 (중심 기준, 픽셀)
            grid_size: 특징 추출 격자 크기 (rows, cols)
        """
        self.roi_size = roi_size
        self.grid_size = grid_size
        self.next_id = 0
        
        # 순차적으로 프레임 저장
        self.frames = []  # [(pan, tilt, timestamp, objects), ...]
        
        # 유사도 로그 (디버깅용)
        self.similarity_log = []
        
        # 고유 ID 카운터 (검출용, 1부터 시작)
        self.unique_id_counter = 1
        
        # ⭐ 병합 로그
        self.merge_log = []  # Track 병합 정보
        
    def reset(self):
        """Tracker 상태 초기화"""
        self.next_id = 0
        self.frames = []
        self.similarity_log = []
        self.unique_id_counter = 1
        self.merge_log = []
        
    def add_detections(self, boxes, scores, img_on, diff, pan, tilt, timestamp):
        """
        타임스탬프 기반 순차 추적:
        1. 직전 프레임 (threshold=0.3)
        2. 프레임 건너뛰기 (threshold=0.35) - 검출 놓침 대비
        ⭐ 헝가리안 알고리즘 (최적 매칭)
        
        Args:
            boxes: [(x, y, w, h), ...] - YOLO 검출 박스
            scores: [conf, ...] - 신뢰도
            img_on: LED ON 이미지
            diff: Diff 이미지
            pan, tilt: 현재 프레임 위치
            timestamp: 타임스탬프
        
        Returns:
            track_ids: [track_id, ...] - 각 박스의 track_id
        """
        # 현재 프레임 특징 추출
        curr_objects = []
        H, W = img_on.shape[:2]
        
        for i, (x, y, w, h) in enumerate(boxes):
            # ⭐ 객체 중심 계산
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            
            # ⭐ 중심 기준 고정 크기 ROI
            half_size = self.roi_size // 2
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(W, center_x + half_size)
            y2 = min(H, center_y + half_size)
            
            roi = img_on[y1:y2, x1:x2]
            diff_roi = diff[y1:y2, x1:x2]  # ⭐ Diff ROI도 추출
            
            if roi.size == 0:
                continue
                
            # ⭐ diff_roi 전달하여 필름 필터링 (grid_size 전달)
            vec = get_feature_vector(roi, diff_roi=diff_roi, grid_size=self.grid_size)
            
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
        
        # 2. 유사도 매트릭스 생성 및 헝가리안 알고리즘 적용
        # ⭐ HUNGARIAN ALGORITHM (최적 매칭)
        all_candidates = direct_candidates + skip_candidates
        
        if not all_candidates:
            # 후보가 없으면 모두 새 ID
            track_ids = []
            for obj_idx, obj in enumerate(curr_objects):
                track_id = self.next_id
                self.next_id += 1
                obj['track_id'] = track_id
                track_ids.append(track_id)
                
                # 로그 생성
                log_entry = {
                    'pan': pan,
                    'tilt': tilt,
                    'timestamp': timestamp,
                    'obj_idx': obj_idx,
                    'unique_id': obj['unique_id'],
                    'comparisons': [],
                    'assigned_id': track_id,
                    'best_similarity': 0.0,
                    'is_new_object': True,
                    'match_source': None
                }
                self.similarity_log.append(log_entry)
        else:
            # ⭐ Cost Matrix 생성 (유사도를 cost로 변환: cost = 1 - similarity)
            n_objects = len(curr_objects)
            n_candidates = len(all_candidates)
            
            # Cost matrix: rows=objects, cols=candidates
            cost_matrix = np.ones((n_objects, n_candidates))  # 기본 cost = 1 (유사도 0)
            similarity_matrix = np.zeros((n_objects, n_candidates))
            
            for i, obj in enumerate(curr_objects):
                for j, candidate in enumerate(all_candidates):
                    sim = calc_cosine_similarity(obj['vec'], candidate['vec'])
                    similarity_matrix[i, j] = sim
                    cost_matrix[i, j] = 1.0 - sim  # cost = 1 - similarity
            
            # ⭐ 헝가리안 알고리즘 적용 (최소 비용 매칭)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # 3. 매칭 결과 적용 (임계값 필터링)
            obj_assignments = {}
            
            for obj_idx, cand_idx in zip(row_ind, col_ind):
                candidate = all_candidates[cand_idx]
                sim = similarity_matrix[obj_idx, cand_idx]
                
                # 후보 소스 판단 (direct vs skip)
                if cand_idx < len(direct_candidates):
                    source = 'direct'
                    threshold = 0.3  # Direct threshold
                else:
                    source = 'skip'
                    threshold = 0.35  # Skip threshold
                
                # ⭐ 임계값 이상일 때만 매칭
                if sim >= threshold:
                    obj_assignments[obj_idx] = (candidate['track_id'], sim, candidate, source)
            
            # 4. 최종 track_id 할당
            track_ids = []
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
        프레임 후보 검색 (⭐ 양방향 대각선 1개씩 버전):
        1. n-1 (최근 1프레임): 같은 Pan, 같은 Tilt, 대각선
        2. n-2 (2프레임 전): 같은 Pan, 같은 Tilt
        3. ⭐ 양방향 대각선: 각 방향에서 1개씩만 수집
        
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
        
        # ⭐ 양방향 대각선 (지그재그 스캔 대응) - 각 방향에서 1개씩만
        diagonal_increase = None  # Pan 증가, Tilt 변화
        diagonal_decrease = None  # Pan 감소, Tilt 변화
        
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
            
            # ⭐ 양방향 대각선 검색 (각 방향에서 1개씩만!)
            if frame_pan != current_pan and frame_tilt != current_tilt:
                # Pan 증가 방향 (→)
                if frame_pan < current_pan and diagonal_increase is None:
                    diagonal_increase = prev_frame
                
                # Pan 감소 방향 (←)
                if frame_pan > current_pan and diagonal_decrease is None:
                    diagonal_decrease = prev_frame
            
            # ⭐ 충분히 수집했으면 종료
            if (skip_pan_frame is not None and skip_tilt_frame is not None and 
                diagonal_increase is not None and diagonal_decrease is not None):
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
        
        # ⭐ 대각선 (조건 없이 항상 추가)
        if prev_diagonal_frame is not None:
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
        
        # ⭐ 양방향 대각선 (각 방향에서 1개씩)
        if diagonal_increase is not None:
            for obj in diagonal_increase['objects']:
                skip_candidates.append({
                    **obj,
                    'frame_pan': diagonal_increase['pan'],
                    'frame_tilt': diagonal_increase['tilt'],
                    'frame_timestamp': diagonal_increase['timestamp']
                })
        
        if diagonal_decrease is not None:
            for obj in diagonal_decrease['objects']:
                skip_candidates.append({
                    **obj,
                    'frame_pan': diagonal_decrease['pan'],
                    'frame_tilt': diagonal_decrease['tilt'],
                    'frame_timestamp': diagonal_decrease['timestamp']
                })
        
        return {'direct': direct_candidates, 'skip': skip_candidates}
    
    def get_track_count(self):
        """할당된 총 track_id 개수 반환"""
        return self.next_id
    
    def get_all_tracks(self):
        """모든 track별 검출 정보 반환 (시각화/분석용)"""
        tracks = {}  # {track_id: [{'pan': , 'tilt': , 'box': , ...}, ...]}
        
        for frame in self.frames:
            for obj in frame['objects']:
                track_id = obj['track_id']
                if track_id not in tracks:
                    tracks[track_id] = []
                
                tracks[track_id].append({
                    'pan': frame['pan'],
                    'tilt': frame['tilt'],
                    'timestamp': frame['timestamp'],
                    'box': obj['box'],
                    'unique_id': obj['unique_id']
                })
        
        return tracks
    
    def merge_similar_tracks(self, merge_threshold=0.4, min_detections=3):
        """
        추적 완료 후 유사한 track들을 병합하는 후처리 단계
        
        Args:
            merge_threshold: Track 병합 임계값 (⭐ 사용자 조정 가능)
            min_detections: 최소 검출 개수 (이보다 적으면 제외)
        
        Returns:
            merge_map: {old_track_id: new_track_id} 매핑
        """
        print(f"\n{'='*60}")
        print(f"🔄 Track 병합 시작 (threshold={merge_threshold}, min={min_detections})")
        print(f"{'='*60}")
        
        # 1. 각 track별 검출 수집
        tracks = {}  # {track_id: [obj1, obj2, ...]}
        for frame in self.frames:
            for obj in frame['objects']:
                track_id = obj['track_id']
                if track_id not in tracks:
                    tracks[track_id] = []
                tracks[track_id].append(obj)
        
        # 2. 모든 Track 처리 (1개 이상)
        all_track_ids = list(tracks.keys())
        valid_tracks = {tid: objs for tid, objs in tracks.items() if len(objs) >= min_detections}
        small_tracks = {tid: objs for tid, objs in tracks.items() if 1 <= len(objs) < min_detections}
        
        print(f"  총 Track 수: {len(tracks)}개")
        print(f"  큰 Track (>= {min_detections}개): {len(valid_tracks)}개")
        print(f"  작은 Track (1~{min_detections-1}개): {len(small_tracks)}개")
        if small_tracks:
            print(f"    작은 Track IDs: {list(small_tracks.keys())}")
        
        # 3. 적응형 샘플링 함수
        def get_adaptive_samples(objs_a, objs_b):
            """
            두 Track의 크기에 맞춰 적응형 샘플링
            작은 쪽 크기에 맞춰 큰 쪽도 샘플링
            """
            n_a = len(objs_a)
            n_b = len(objs_b)
            n_samples = min(n_a, n_b, 3)  # 최대 3개
            
            # Track A 샘플링
            if n_a >= 3 and n_samples == 3:
                idx_a = [0, n_a // 2, n_a - 1]
            elif n_a == 2 and n_samples == 2:
                idx_a = [0, 1]
            elif n_a == 1 or n_samples == 1:
                idx_a = [0]
            else:
                idx_a = list(range(min(n_a, n_samples)))
            
            # Track B 샘플링
            if n_b >= 3 and n_samples == 3:
                idx_b = [0, n_b // 2, n_b - 1]
            elif n_b == 2 and n_samples == 2:
                idx_b = [0, 1]
            elif n_b == 1 or n_samples == 1:
                idx_b = [0]
            else:
                idx_b = list(range(min(n_b, n_samples)))
            
            samples_a = [objs_a[i] for i in idx_a]
            samples_b = [objs_b[i] for i in idx_b]
            
            return samples_a, samples_b, n_samples
        
        # 4. Track 간 유사도 계산 및 병합 그룹 생성
        merge_groups = []  # [[tid1, tid2, ...], ...]
        visited = set()
        
        # ⭐ 병합 비교 로그
        comparison_log = []
        
        # ⭐ 새로운 방식: frames에서 track별 위치 정보 수집
        track_by_position = {}  # {track_id: [(pan, tilt, obj), ...]}
        for frame in self.frames:
            pan = frame['pan']
            tilt = frame['tilt']
            for obj in frame['objects']:
                tid = obj['track_id']
                if tid not in track_by_position:
                    track_by_position[tid] = []
                track_by_position[tid].append((pan, tilt, obj))
        
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
                
                # ⭐ 공간 기반 샘플링: 같은 Pan 라인 또는 같은 Tilt 라인만 비교
                similarities = []
                
                # 같은 Pan 라인
                for pan_a, tilt_a, obj_a in positions_a:
                    for pan_b, tilt_b, obj_b in positions_b:
                        if pan_a == pan_b:  # 같은 Pan 라인
                            sim = calc_cosine_similarity(obj_a['vec'], obj_b['vec'])
                            similarities.append(sim)
                
                # 같은 Tilt 라인
                for pan_a, tilt_a, obj_a in positions_a:
                    for pan_b, tilt_b, obj_b in positions_b:
                        if tilt_a == tilt_b:  # 같은 Tilt 라인
                            sim = calc_cosine_similarity(obj_a['vec'], obj_b['vec'])
                            similarities.append(sim)
                
                if not similarities:
                    # 겹치는 라인이 없으면 비교 불가
                    continue
                
                avg_sim = np.mean(similarities)
                min_sim = np.min(similarities)
                max_sim = np.max(similarities)
                
                # ⭐ 비교 로그 기록
                comparison_log.append({
                    'track_a': tid_a,
                    'track_a_count': len(positions_a),
                    'track_b': tid_b,
                    'track_b_count': len(positions_b),
                    'samples_used': len(similarities),
                    'avg_similarity': float(avg_sim),
                    'min_similarity': float(min_sim),
                    'max_similarity': float(max_sim),
                    'num_comparisons': len(similarities),
                    'similarities': [float(s) for s in similarities],  # ⭐ 개별 값 저장
                    'merged': avg_sim >= merge_threshold
                })
                
                # ⭐ 임계값 이상이면 같은 그룹으로 병합
                if avg_sim >= merge_threshold:
                    print(f"  ✅ Track {tid_a}({len(positions_a)}개) ↔ Track {tid_b}({len(positions_b)}개): "
                          f"유사도 {avg_sim:.4f} (비교 {len(similarities)}회) → 병합!")
                    group.append(tid_b)
                    visited.add(tid_b)
            
            merge_groups.append(group)
        
        # 5. 병합 맵 생성 (가장 작은 ID를 대표로 사용)
        merge_map = {}
        for group in merge_groups:
            representative_id = min(group)  # 가장 작은 ID를 대표로
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
        print(f"    최종 Track 수: {len(merge_groups)}개 (유효 Track 중)")
        print(f"    병합된 검출: {merged_count}개")
        print(f"{'='*60}\n")
        
        # 7. 병합 로그 저장
        self.merge_log = {
            'threshold': merge_threshold,
            'min_detections': min_detections,
            'total_tracks': len(tracks),
            'valid_tracks': len(valid_tracks),
            'excluded_tracks': {tid: len(objs) for tid, objs in small_tracks.items()},  # ⭐ small_tracks로 변경
            'final_tracks': len(merge_groups),
            'merged_count': merged_count,
            'comparisons': comparison_log,
            'merge_groups': merge_groups
        }
        
        return merge_map
