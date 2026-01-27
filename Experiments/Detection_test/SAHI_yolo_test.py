import os
import cv2
import torch
import pathlib
import numpy as np
from ultralytics import YOLO
from tkinter import Tk, filedialog

def non_max_suppression(boxes, scores, iou_threshold):
    # OpenCV NMS 사용
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

def predict_with_tiling(model, img, rows=2, cols=3, overlap=0.15, conf=0.25, iou=0.45, device='cuda', use_full_image=True, nms_iou=0.3, nms_iomin=0.5):
    """
    이미지를 타일로 쪼개서 예측 후 결과 병합
    rows, cols: 행/열 개수 (2x3 = 6등분)
    overlap: 타일 간 겹치는 비율 (0.15 = 15%)
    use_full_image: 전체 이미지도 함께 검출 (큰 객체 검출용)
    nms_iou: NMS IoU threshold (타일 병합 후)
    nms_iomin: NMS IoMin threshold (중첩 박스 제거)
    """
    H, W = img.shape[:2]
    
    # ⭐ yolo_utils.py와 동일한 타일 생성 방식
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

    # 모든 타일 추론
    all_boxes = []   # [x, y, w, h]
    all_scores = []
    all_classes = []
    
    total_images = len(tiles) + (1 if use_full_image else 0)
    print(f"  -> {total_images}개 이미지 분석 중 (전체 이미지: {use_full_image}, 타일: {len(tiles)}개)...", end="", flush=True)
    
    # ⭐ 1. 전체 이미지 검출 (옵션)
    if use_full_image:
        results = model.predict(img, conf=conf, iou=iou, device=device, verbose=False)[0]
        if results.boxes:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                c = float(box.conf.cpu().numpy().item())
                cls = int(box.cls.cpu().numpy().item())
                
                w = x2 - x1
                h = y2 - y1
                
                all_boxes.append([int(x1), int(y1), int(w), int(h)])
                all_scores.append(c)
                all_classes.append(cls)
        print("F", end="", flush=True)  # Full image 완료
    
    # ⭐ 2. 타일 검출
    for i, (tx1, ty1, tx2, ty2) in enumerate(tiles):
        # 타일 잘라내기
        tile_img = img[ty1:ty2, tx1:tx2]
        
        # YOLO 추론
        results = model.predict(tile_img, conf=conf, iou=iou, device=device, verbose=False)[0]
        
        if results.boxes:
            for box in results.boxes:
                # 로컬 좌표
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                c = float(box.conf.cpu().numpy().item())
                cls = int(box.cls.cpu().numpy().item())
                
                # 글로벌 좌표로 변환
                gx1 = x1 + tx1
                gy1 = y1 + ty1
                gx2 = x2 + tx1
                gy2 = y2 + ty1
                
                # NMS를 위해 [x, y, w, h] 형식으로 저장
                w = gx2 - gx1
                h = gy2 - gy1
                
                all_boxes.append([int(gx1), int(gy1), int(w), int(h)])
                all_scores.append(c)
                all_classes.append(cls)
        print(".", end="", flush=True)
    print(" 완료")

    # ⭐ 3. IoU + IoMin NMS 수행 (중복 박스 + 중첩 박스 제거)
    print(f"  -> NMS 전: {len(all_boxes)}개 박스", end="", flush=True)
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
    
    print(f" -> NMS 후: {len(final_boxes)}개 박스")
    return final_boxes, final_scores, final_classes

def draw_results(img, boxes, scores, classes):
    """결과 그리기"""
    plot_img = img.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        x1, y1 = int(x), int(y)
        x2, y2 = int(x+w), int(y+h)
        score = scores[i]
        cls = classes[i]
        
        # 박스 그리기
        cv2.rectangle(plot_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # 라벨
        label = f"{score:.2f}"
        cv2.putText(plot_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return plot_img

def run_yolo_on_folder():
    # 1. Tkinter 초기화
    root = Tk()
    root.withdraw()

    print("=== YOLO 타일링(분할) 분석 도구 ===")
    print("이미지를 6등분(2x3)하여 작은 객체를 정밀 탐지합니다.")

    # 2. YOLO 가중치 파일 선택
    # print("\n[1] YOLO 가중치 파일(.pt)을 선택하세요...")
    # model_path = filedialog.askopenfilename(
    #     title="YOLO 가중치 파일 선택",
    #     filetypes=[("YOLO weights", "*.pt"), ("All files", "*.*")]
    # )
    # if not model_path: return
    
    # 편의를 위해 경로 고정 (필요시 주석 해제하여 선택창 사용)
    model_path = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\PTCamera_waveshare\yolov11m_diff.pt"
    if not os.path.exists(model_path):
         model_path = filedialog.askopenfilename(title="YOLO 가중치 파일 선택")

    print(f"-> 모델: {model_path}")

    # 3. 테스트할 이미지 폴더 선택
    print("\n[2] 테스트할 이미지가 있는 폴더를 선택하세요...")
    target_dir = filedialog.askdirectory(title="이미지 폴더 선택")
    if not target_dir: return
    target_path = pathlib.Path(target_dir)

    # 4. 결과 저장 폴더
    save_dir = target_path / "yolo_tiled_results"
    save_dir.mkdir(exist_ok=True)

    # 5. 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[3] 모델 로딩 중... (Device: {device})")
    model = YOLO(model_path)

    # 6. 이미지 파일 찾기
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(target_path.glob(ext))
        files.extend(target_path.glob(ext.upper()))
    files = sorted(list(set(files)))
    
    if not files:
        print("이미지가 없습니다.")
        return

    print(f"\n[4] 총 {len(files)}장 처리 시작 (2x3 분할)...")

    # 7. 실행
    count = 0
    for i, fpath in enumerate(files):
        try:
            img = cv2.imread(str(fpath))
            if img is None: continue

            # ★ 타일링 추론 실행 ★
            # ⭐ MOT와 동일한 설정
            # rows=2, cols=3 -> 6등분
            # overlap=0.15 -> 15% (MOT와 동일)
            # conf=0.50 -> MOT CONF_THRES와 동일
            # iou=0.2 -> MOT IOU_THRES와 동일
            # use_full_image=True -> 전체 이미지도 함께 검출 (7개 이미지)
            boxes, scores, classes = predict_with_tiling(
                model, img, 
                rows=2, cols=3, 
                overlap=0.15,  # ⭐ MOT와 동일
                conf=0.50,     # ⭐ MOT와 동일
                iou=0.2,       # ⭐ MOT와 동일
                device=device,
                use_full_image=True
            )

            # 결과 그리기
            res_img = draw_results(img, boxes, scores, classes)

            # 저장
            save_path = save_dir / f"res_{fpath.name}"
            cv2.imwrite(str(save_path), res_img)
            
            count += 1
            print(f"[{i+1}/{len(files)}] {fpath.name} -> {len(boxes)}개 검출")

        except Exception as e:
            print(f"에러 ({fpath.name}): {e}")
            import traceback
            traceback.print_exc()

    print(f"\n=== 완료! 저장 위치: {save_dir} ===")

if __name__ == "__main__":
    run_yolo_on_folder()
