#!/usr/bin/env python3
"""
Multi-Object Tracking (MOT) Module
타임스탬프 기반 순차 추적 알고리즘
HSV + Grayscale 히스토그램 특징 추출 및 코사인 유사도 기반 매칭

Modified for PTCamera_waveshare:
- Grid: 11x11 (세밀한 특징, 5808차원)
- Histogram: HSV [8,4] + Gray [16] = 48차원/cell (hungarian_final과 동일)
- Threshold: 0.5 (모든 후보 동일, 엄격한 매칭)
- Image: LED OFF for feature extraction
"""

import cv2
import os
import re
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
        roi_bgr: BGR 이미지 ROI (LED OFF)
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
    
    Modified thresholds:
    - Direct: 0.5 (was 0.3)
    - Skip: 0.5 (was 0.35)
    """
    
    def __init__(self, roi_size=300, grid_size=(11, 11)):
        """
        Args:
            roi_size: 고정 ROI 크기 (중심 기준, 픽셀)
            grid_size: 특징 추출 격자 크기 (rows, cols) - 기본 11x11 (5808차원)
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
        1. 직전 프레임 (threshold=0.5)
        2. 프레임 건너뛰기 (threshold=0.5) - 검출 놓침 대비
        ⭐ 헝가리안 알고리즘 (최적 매칭)
        
        Args:
            boxes: [(x, y, w, h), ...] - YOLO 검출 박스
            scores: [conf, ...] - 신뢰도
            img_on: LED OFF 이미지 (특징 추출용)
            diff: Diff 이미지 (마스크용)
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
                
                # ⭐ 모든 후보에 대해 동일한 threshold (0.4) 적용
                threshold = 0.4
                
                # 후보 소스 판단 (로깅용)
                if cand_idx < len(direct_candidates):
                    source = 'direct'
                else:
                    source = 'skip'
                
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
                
                # ⭐ 로그 생성 (모든 후보와의 비교 기록 - mot_scan_test_hungarian_final.py와 동일)
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
        프레임 후보 검색 (⭐ 5개 후보군 - 완전 공간 검색 버전):
        1. n-1 (최근 1프레임)
        2. n-2 (2프레임 전)
        3. 바로 윗줄(Previous Row)에서 공간적으로 가장 가까운 3개:
           - Vertical (Pan == Current)
           - Diag Left (Pan < Current 중 최대)
           - Diag Right (Pan > Current 중 최소)
        
        반환: {'direct': [...], 'skip': [...]}
        """
        if not self.frames:
            return {'direct': [], 'skip': []}
        
        # 1. n-1 & 2. n-2 프레임 (시간적 이웃)
        frame_n1 = self.frames[-1] if len(self.frames) >= 1 else None
        frame_n2 = self.frames[-2] if len(self.frames) >= 2 else None
        
        # 3. 바로 윗줄(Previous Row) 프레임들 수집
        prev_row_frames = []
        target_prev_tilt = None
        
        # 역순 탐색으로 바로 윗줄(Tilt가 다른 첫 줄)을 찾음
        for i in range(len(self.frames)):
            f = self.frames[-(i+1)]
            
            # 현재 줄(Tilt 같음)은 패스 (n-1, n-2에서 이미 처리됨)
            if f['tilt'] == current_tilt:
                continue
            
            # 윗줄 발견
            if target_prev_tilt is None:
                target_prev_tilt = f['tilt']
            
            # 같은 윗줄이면 수집
            if f['tilt'] == target_prev_tilt:
                prev_row_frames.append(f)
            else:
                # 더 윗줄(Previous Previous Row)이 나오면 종료
                break
        
        # 수집된 윗줄 프레임 중에서 공간적으로 가장 가까운 3개 찾기
        vertical_frame = None       # Pan == Current
        diagonal_increase = None    # Pan < Current (Left)
        diagonal_decrease = None    # Pan > Current (Right)
        
        # 거리가 가장 가까운 놈을 찾기 위한 변수
        min_dist_inc = float('inf')
        min_dist_dec = float('inf')
        
        for f in prev_row_frames:
            f_pan = f['pan']
            
            # Vertical
            if f_pan == current_pan:
                vertical_frame = f
            
            # Diag Left (Pan < Current)
            elif f_pan < current_pan:
                dist = current_pan - f_pan
                if dist < min_dist_inc:
                    min_dist_inc = dist
                    diagonal_increase = f
            
            # Diag Right (Pan > Current)
            elif f_pan > current_pan:
                dist = f_pan - current_pan
                if dist < min_dist_dec:
                    min_dist_dec = dist
                    diagonal_decrease = f
        
        # 후보군 수집
        direct_candidates = []
        skip_candidates = []
        
        # 1. n-1 (Direct)
        if frame_n1:
            for obj in frame_n1['objects']:
                direct_candidates.append({
                    **obj,
                    'frame_pan': frame_n1['pan'],
                    'frame_tilt': frame_n1['tilt'],
                    'frame_timestamp': frame_n1['timestamp']
                })
        
        # 나머지 4개는 Skip 후보로 통합
        candidates_frames = []
        if frame_n2: candidates_frames.append(frame_n2)
        if vertical_frame: candidates_frames.append(vertical_frame)
        if diagonal_increase: candidates_frames.append(diagonal_increase)
        if diagonal_decrease: candidates_frames.append(diagonal_decrease)
        
        # 중복 제거 (n-1과 track_id 기준)
        added_track_ids = set()
        for cand in direct_candidates:
            added_track_ids.add(cand['track_id'])
            
        for frame in candidates_frames:
            if frame is frame_n1: continue
            
            for obj in frame['objects']:
                if obj['track_id'] in added_track_ids:
                    continue
                
                skip_candidates.append({
                    **obj,
                    'frame_pan': frame['pan'],
                    'frame_tilt': frame['tilt'],
                    'frame_timestamp': frame['timestamp']
                })
                added_track_ids.add(obj['track_id'])

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
            merge_threshold: Track 병합 임계값
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
        
        # 2. 모든 Track 처리
        all_track_ids = list(tracks.keys())
        valid_tracks = {tid: objs for tid, objs in tracks.items() if len(objs) >= min_detections}
        small_tracks = {tid: objs for tid, objs in tracks.items() if 1 <= len(objs) < min_detections}
        
        print(f"  총 Track 수: {len(tracks)}개")
        print(f"  큰 Track (>= {min_detections}개): {len(valid_tracks)}개")
        print(f"  작은 Track (1~{min_detections-1}개): {len(small_tracks)}개")
        if small_tracks:
            print(f"    작은 Track IDs: {list(small_tracks.keys())}")
        
        # 3. Track별 위치 정보 수집
        track_by_position = {}  # {track_id: [(pan, tilt, obj), ...]}
        for frame in self.frames:
            pan = frame['pan']
            tilt = frame['tilt']
            for obj in frame['objects']:
                tid = obj['track_id']
                if tid not in track_by_position:
                    track_by_position[tid] = []
                track_by_position[tid].append((pan, tilt, obj))
        
        # 4. Track 간 유사도 계산 및 병합 그룹 생성
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
                
                # 같은 Pan/Tilt 라인만 비교
                similarities = []
                
                for pan_a, tilt_a, obj_a in positions_a:
                    for pan_b, tilt_b, obj_b in positions_b:
                        if pan_a == pan_b or tilt_a == tilt_b:
                            sim = calc_cosine_similarity(obj_a['vec'], obj_b['vec'])
                            similarities.append(sim)
                
                if not similarities:
                    continue
                
                avg_sim = np.mean(similarities)
                
                # 임계값 이상이면 병합
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
        
        # 7. 병합 로그 저장 (메모리)
        self.merge_log = {
            'threshold': merge_threshold,
            'min_detections': min_detections,
            'total_tracks': len(tracks),
            'valid_tracks': len(valid_tracks),
            'excluded_tracks': {tid: len(objs) for tid, objs in small_tracks.items()},
            'final_tracks': len(merge_groups),
            'merged_count': merged_count,
            'comparisons': [], # merge_log에 comparisons가 없으므로 빈 리스트 (실시간 로깅 특성상 생략 가능하거나 추가 필요)
            'merge_groups': merge_groups
        }
        
        return merge_map

    def save_similarity_log(self, output_path="similarity_log_live.txt"):
        """유사도 로그를 텍스트 파일로 저장 (mot_scan_test_hungarian_final.py와 동일 포맷)"""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("MOT Similarity Log (Live Scan Version)\n")
                f.write("=" * 80 + "\n\n")
                
                for entry in self.similarity_log:
                    f.write(f"\n[Frame] Pan={entry['pan']:+4d}, Tilt={entry['tilt']:+3d}, "
                           f"Timestamp={entry['timestamp']}, Object #{entry['obj_idx']}\n")
                    f.write(f"  🆔 Unique ID: {entry['unique_id']}\n")
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
                            cand_uniq = comp.get('candidate_unique_id', 'N/A')
                            f.write(f"{marker} Track ID {comp['candidate_id']:3d} "
                                   f"[{cand_uniq}] "
                                   f"(Pan={comp['candidate_pan']:+4d}, Tilt={comp['candidate_tilt']:+3d}) "
                                   f"→ Sim: {comp['similarity']:.4f}\n")
                    else:
                        f.write(f"  ℹ️  No candidates (first detection)\n")
                    
                    f.write("-" * 80 + "\n")
                
                # 병합 로그 추가
                if self.merge_log:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("🔄 Track 병합 정보 (Post-Processing)\n")
                    f.write("=" * 80 + "\n\n")
                    
                    f.write(f"병합 설정:\n")
                    f.write(f"  - Threshold: {self.merge_log.get('threshold', 'N/A')}\n")
                    f.write(f"  - Min Detections: {self.merge_log.get('min_detections', 'N/A')}\n\n")
                    
                    f.write(f"통계:\n")
                    f.write(f"  - 총 Track 수: {self.merge_log.get('total_tracks', 0)}개\n")
                    f.write(f"  - 유효 Track: {self.merge_log.get('valid_tracks', 0)}개\n")
                    excluded = self.merge_log.get('excluded_tracks', {})
                    f.write(f"  - 제외 Track: {len(excluded)}개\n")
                    f.write(f"  - 최종 Track: {self.merge_log.get('final_tracks', 0)}개\n")
                    f.write(f"  - 병합된 검출: {self.merge_log.get('merged_count', 0)}개\n\n")
                    
                    if excluded:
                        f.write(f"제외된 Track IDs:\n")
                        for tid, count in excluded.items():
                            f.write(f"  - Track {tid}: {count}개\n")
                        f.write("\n")
                    
                    f.write("=" * 80 + "\n")
                    # (실시간 버전은 병합 상세 로그 생략 또는 추후 추가)
                    # merge_log에 comparisons 리스트가 없으므로 여기까지만 출력
        except Exception as e:
            print(f"❌ 로그 저장 실패: {e}")
