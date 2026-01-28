#!/usr/bin/env python3
"""
YOLO object detection utilities
Handles model loading, tiled inference, and NMS
"""

import cv2
import numpy as np

# YOLO support
try:
    from ultralytics import YOLO
    _YOLO_OK = True
except Exception:
    YOLO = None
    _YOLO_OK = False

# Torch support
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    _TORCH_AVAILABLE = False


def non_max_suppression(boxes, scores, iou_threshold):
    """Non-Maximum Suppression using OpenCV"""
    if len(boxes) == 0:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    if len(indices) > 0:
        return indices.flatten()
    return []


def improved_nms(boxes, scores, iou_threshold=0.3, io_min_threshold=0.5):
    """
    IoU + IoMin 결합 NMS
    - IoU: 일반적인 중복 제거
    - IoMin: 작은 박스가 큰 박스에 포함된 경우 제거
    
    IoMin = Intersection / Min(Area1, Area2)
    → 작은 박스가 큰 박스에 50% 이상 포함되면 제거
    """
    if len(boxes) == 0:
        return []
    
    # 1. 먼저 일반 IoU 기반 NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    if len(indices) == 0:
        return []
    
    indices = indices.flatten().tolist()
    
    # 2. IoMin으로 중첩된 박스 추가 제거
    keep = []
    for i in indices:
        should_keep = True
        box_i = boxes[i]
        area_i = box_i[2] * box_i[3]
        
        for j in keep:
            box_j = boxes[j]
            area_j = box_j[2] * box_j[3]
            
            # Intersection 계산
            x1_i, y1_i = box_i[0], box_i[1]
            x2_i, y2_i = box_i[0] + box_i[2], box_i[1] + box_i[3]
            x1_j, y1_j = box_j[0], box_j[1]
            x2_j, y2_j = box_j[0] + box_j[2], box_j[1] + box_j[3]
            
            inter_x1 = max(x1_i, x1_j)
            inter_y1 = max(y1_i, y1_j)
            inter_x2 = min(x2_i, x2_j)
            inter_y2 = min(y2_i, y2_j)
            
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                
                # ⭐ IoMin = Intersection / Min(area_i, area_j)
                io_min = inter_area / min(area_i, area_j)
                
                if io_min > io_min_threshold:  # 50% 이상 포함되면 제거
                    should_keep = False
                    break
        
        if should_keep:
            keep.append(i)
    
    return keep



def improved_nms(boxes, scores, iou_threshold=0.3, io_min_threshold=0.5):
    """
    IoU + IoMin 결합 NMS
    - IoU: 일반적인 중복 제거
    - IoMin: 작은 박스가 큰 박스에 포함된 경우 제거
    
    IoMin = Intersection / Min(Area1, Area2)
    → 작은 박스가 큰 박스에 50% 이상 포함되면 제거
    """
    if len(boxes) == 0:
        return []
    
    # 1. 먼저 일반 IoU 기반 NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    if len(indices) == 0:
        return []
    
    indices = indices.flatten().tolist()
    
    # 2. IoMin으로 중첩된 박스 추가 제거
    keep = []
    for i in indices:
        should_keep = True
        box_i = boxes[i]
        area_i = box_i[2] * box_i[3]
        
        for j in keep:
            box_j = boxes[j]
            area_j = box_j[2] * box_j[3]
            
            # Intersection 계산
            x1_i, y1_i = box_i[0], box_i[1]
            x2_i, y2_i = box_i[0] + box_i[2], box_i[1] + box_i[3]
            x1_j, y1_j = box_j[0], box_j[1]
            x2_j, y2_j = box_j[0] + box_j[2], box_j[1] + box_j[3]
            
            inter_x1 = max(x1_i, x1_j)
            inter_y1 = max(y1_i, y1_j)
            inter_x2 = min(x2_i, x2_j)
            inter_y2 = min(y2_i, y2_j)
            
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                
                # ⭐ IoMin = Intersection / Min(area_i, area_j)
                io_min = inter_area / min(area_i, area_j)
                
                if io_min > io_min_threshold:  # 50% 이상 포함되면 제거
                    should_keep = False
                    break
        
        if should_keep:
            keep.append(i)
    
    return keep


def predict_with_tiling(model, img, rows=2, cols=3, overlap=0.15, conf=0.25, iou=0.45, device='cuda', use_full_image=True, nms_iou=0.3, nms_iomin=0.5):
    """
    이미지를 타일로 쪼개서 예측 후 결과 병합
    rows, cols: 행/열 개수 (2x3 = 6등분)
    overlap: 타일 간 겹치는 비율 (0.15 = 15%)
    use_full_image: 전체 이미지도 함께 검출 (큰 객체 검출용)
    nms_iou: NMS IoU threshold (타일 병합 후)
    nms_iomin: NMS IoMin threshold (중첩 박스 제거)
    
    배치 처리: 전체 이미지 + 타일 6개를 한 번에!
    """
    H, W = img.shape[:2]
    
    # 타일 크기 계산 (겹침 포함)
    tile_h = int(H / rows)
    tile_w = int(W / cols)
    
    # 겹침 크기
    ov_h = int(tile_h * overlap)
    ov_w = int(tile_w * overlap)
    
    # 실제 타일 크기 (겹침 포함)
    step_h = tile_h - ov_h
    
    # 타일 좌표 생성 (정확히 rows x cols 개수)
    tiles = []
    base_tile_h = H // rows
    base_tile_w = W // cols
    
    # 오버랩 크기 계산
    ov_h = int(base_tile_h * overlap)
    ov_w = int(base_tile_w * overlap)
    
    for row_idx in range(rows):
        for col_idx in range(cols):
            # 기본 타일 영역
            y1 = row_idx * base_tile_h
            y2 = (row_idx + 1) * base_tile_h if row_idx < rows - 1 else H
            x1 = col_idx * base_tile_w
            x2 = (col_idx + 1) * base_tile_w if col_idx < cols - 1 else W
            
            # 오버랩 확장 (경계 체크)
            y1 = max(0, y1 - ov_h)
            y2 = min(H, y2 + ov_h)
            x1 = max(0, x1 - ov_w)
            x2 = min(W, x2 + ov_w)
            
            tiles.append((x1, y1, x2, y2))
    
    print(f"[DEBUG] 타일 개수: {len(tiles)}, 이미지 크기: {W}x{H}, rows={rows}, cols={cols}")

    # 모든 이미지 수집 (배치 처리용)
    all_boxes = []
    all_scores = []
    all_classes = []
    
    # 배치 크기 자동 조정
    # 전체 이미지 포함 시: 7개, 3개, 1개
    # 타일만: 6개, 3개, 1개
    total_images = len(tiles) + (1 if use_full_image else 0)
    batch_sizes = [total_images, 3, 1]
    
    for batch_size in batch_sizes:
        try:
            # 배치 이미지 준비
            all_batch_images = []
            image_info = []  # (type, tx1, ty1, tx2, ty2) - type: 'full' or 'tile'
            
            # 1. 전체 이미지 먼저 (큰 객체 검출용)
            if use_full_image:
                all_batch_images.append(img)
                image_info.append(('full', 0, 0, W, H))
            
            # 2. 타일 이미지들
            for tx1, ty1, tx2, ty2 in tiles:
                tile_img = img[ty1:ty2, tx1:tx2]
                all_batch_images.append(tile_img)
                image_info.append(('tile', tx1, ty1, tx2, ty2))
            
            # 배치로 처리
            for i in range(0, len(all_batch_images), batch_size):
                batch_images = all_batch_images[i:i + batch_size]
                batch_info = image_info[i:i + batch_size]
                
                # YOLO 배치 추론
                if len(batch_images) == 1:
                    results_list = [model.predict(batch_images[0], conf=conf, iou=iou, device=device, verbose=False)[0]]
                else:
                    results_list = model.predict(batch_images, conf=conf, iou=iou, device=device, verbose=False)
                
                # 결과 처리
                for j, (img_type, tx1, ty1, tx2, ty2) in enumerate(batch_info):
                    results = results_list[j]
                    
                    if results.boxes:
                        for box in results.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            c = float(box.conf.cpu().numpy().item())
                            cls = int(box.cls.cpu().numpy().item())
                            
                            # 글로벌 좌표로 변환
                            if img_type == 'full':
                                # 전체 이미지는 그대로
                                gx1, gy1, gx2, gy2 = x1, y1, x2, y2
                            else:
                                # 타일은 오프셋 추가
                                gx1 = x1 + tx1
                                gy1 = y1 + ty1
                                gx2 = x2 + tx1
                                gy2 = y2 + ty1
                            
                            w = gx2 - gx1
                            h = gy2 - gy1
                            
                            all_boxes.append([int(gx1), int(gy1), int(w), int(h)])
                            all_scores.append(c)
                            all_classes.append(cls)
            
            # 성공하면 빠져나오기
            if batch_size == total_images:
                if use_full_image:
                    print(f"[YOLO] 전체+타일 배치 처리 성공: {total_images}개 동시 처리")
                else:
                    print(f"[YOLO] 타일 배치 처리 성공: {batch_size}개 동시 처리")
            elif batch_size == 3:
                print(f"[YOLO] 배치 크기 감소: {batch_size}개씩 처리")
            break
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'cuda' in str(e).lower():
                # 메모리 부족 - 다음 배치 크기 시도
                if batch_size == batch_sizes[-1]:
                    print(f"[YOLO] 메모리 부족: 모든 배치 크기 실패")
                    raise
                else:
                    print(f"[YOLO] 메모리 부족: 배치 크기 {batch_size} 실패, {batch_sizes[batch_sizes.index(batch_size)+1]}로 재시도...")
                    all_boxes = []
                    all_scores = []
                    all_classes = []
                    continue
            else:
                raise

    # 전체 결과에 대해 NMS 수행 (중복 제거)
    if not all_boxes:
        return [], [], []

    indices = improved_nms(all_boxes, all_scores, iou_threshold=nms_iou, io_min_threshold=nms_iomin)
    
    final_boxes = []
    final_scores = []
    final_classes = []
    
    for idx in indices:
        final_boxes.append(all_boxes[idx])
        final_scores.append(all_scores[idx])
        final_classes.append(all_classes[idx])
        
    return final_boxes, final_scores, final_classes


class YOLOProcessor:
    """Handles YOLO model loading and caching"""
    
    def __init__(self):
        self._cached_model = None
        self._cached_path = None
    
    def get_model(self, weights_path):
        """Get YOLO model with caching"""
        if not _YOLO_OK:
            return None
        
        # Return cached model if path matches
        if self._cached_model is not None and self._cached_path == weights_path:
            return self._cached_model
        
        # Load new model
        try:
            self._cached_model = YOLO(weights_path)
            self._cached_path = weights_path
            print(f"[YOLOProcessor] Loaded model: {weights_path}")
            return self._cached_model
        except Exception as e:
            print(f"[YOLOProcessor] Load failed: {e}")
            return None
    
    def get_device(self):
        """Get available device for inference"""
        if not _TORCH_AVAILABLE:
            return "cpu"
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
