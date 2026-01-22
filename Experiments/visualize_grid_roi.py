"""
발표자료용: 11x11 격자 기반 ROI 시각화
객체 인식 → ROI 추출 → 11x11 격자선 표시
"""
import cv2
import numpy as np
import sys
import os
from pathlib import Path
from ultralytics import YOLO

# ---------------------------------------------------------
# 기존 모듈 로드
# ---------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Com'))

try:
    from yolo_utils import predict_with_tiling
    print("✅ yolo_utils 로드 성공!")
except ImportError:
    print("❌ 오류: Com/yolo_utils.py를 찾을 수 없습니다.")
    sys.exit()

# =========================================================
# [설정]
# =========================================================
MODEL_PATH = "yolov11m_diff.pt"
CONF_THRES = 0.50
IOU_THRES = 0.45
ROI_SIZE = 300  # 300x300 픽셀 (고정)
GRID_SIZE = (11, 11)  # 11x11 격자

# ⭐ 자동으로 스캔 폴더에서 이미지 찾기
SCAN_FOLDER = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\PTCamera_waveshare\captures_gui_20260107_181822"

# 출력 폴더
OUTPUT_FOLDER = "./grid_visualization_output"

def draw_grid_on_roi(roi_img, grid_size=(11, 11), color=(0, 255, 0), thickness=1):
    """
    ROI에 격자선 그리기
    
    Args:
        roi_img: ROI 이미지
        grid_size: (rows, cols) - 격자 구역 수
        color: 격자선 색상 (BGR)
        thickness: 선 두께
    
    Returns:
        격자선이 그려진 이미지
    """
    result = roi_img.copy()
    h, w = result.shape[:2]
    rows, cols = grid_size
    
    # 세로선 그리기
    for c in range(cols + 1):
        x = int((c / cols) * w)
        cv2.line(result, (x, 0), (x, h), color, thickness)
    
    # 가로선 그리기
    for r in range(rows + 1):
        y = int((r / rows) * h)
        cv2.line(result, (0, y), (w, y), color, thickness)
    
    return result

def main():
    """메인 실행 함수"""
    print("="*60)
    print("🎯 11x11 격자 ROI 시각화 시작")
    print("="*60)
    
    # 출력 폴더 생성
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 1. 스캔 폴더에서 이미지 자동 검색
    if not os.path.exists(SCAN_FOLDER):
        print(f"❌ 오류: 스캔 폴더를 찾을 수 없습니다: {SCAN_FOLDER}")
        return
    
    # .jpg, .png 파일 찾기
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(SCAN_FOLDER).glob(ext))
    
    if not image_files:
        print(f"❌ 오류: 스캔 폴더에 이미지가 없습니다: {SCAN_FOLDER}")
        return
    
    # 첫 번째 이미지 사용
    test_image_path = str(image_files[0])
    print(f"📁 스캔 폴더: {SCAN_FOLDER}")
    print(f"   총 {len(image_files)}개 이미지 발견")
    print(f"   사용할 이미지: {Path(test_image_path).name}")
    
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"❌ 오류: 이미지를 읽을 수 없습니다: {test_image_path}")
        return
    
    print(f"✅ 이미지 로드 완료")
    print(f"   크기: {img.shape[1]} x {img.shape[0]}")
    
    # 2. YOLO 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 오류: 모델을 찾을 수 없습니다: {MODEL_PATH}")
        return
    
    model = YOLO(MODEL_PATH)
    print(f"✅ YOLO 모델 로드: {MODEL_PATH}")
    
    # 3. 객체 검출
    print("\n🔍 객체 검출 중...")
    results = model.predict(
        img,
        conf=CONF_THRES,
        iou=IOU_THRES,
        verbose=False
    )
    
    # 검출 결과 파싱
    boxes = []
    scores = []
    
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                # (x, y, w, h) 형식으로 변환
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                boxes.append((x, y, w, h))
                scores.append(conf)
    
    print(f"✅ 검출 완료: {len(boxes)}개 객체 발견")
    
    if len(boxes) == 0:
        print("❌ 검출된 객체가 없습니다.")
        return
    
    # 4. 각 검출 결과에 대해 ROI + 격자 시각화
    H, W = img.shape[:2]
    
    for i, (x, y, w, h) in enumerate(boxes):
        print(f"\n{'='*60}")
        print(f"📦 객체 #{i+1}/{len(boxes)} 처리 중...")
        print(f"{'='*60}")
        
        # 객체 중심 계산
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        
        # 중심 기준 고정 크기 ROI
        half_size = ROI_SIZE // 2
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(W, center_x + half_size)
        y2 = min(H, center_y + half_size)
        
        # ROI 추출
        roi = img[y1:y2, x1:x2].copy()
        
        if roi.size == 0:
            print(f"  ⚠️ ROI가 비어있습니다. 건너뜁니다.")
            continue
        
        print(f"  ✅ ROI 추출: {roi.shape[1]} x {roi.shape[0]}")
        print(f"     중심: ({center_x}, {center_y})")
        print(f"     좌표: ({x1}, {y1}) → ({x2}, {y2})")
        
        # 5. 격자선 그리기
        roi_with_grid = draw_grid_on_roi(roi, grid_size=GRID_SIZE, color=(0, 255, 0), thickness=2)
        
        # 6. 정보 텍스트 추가
        info_text = f"11x11 Grid | ROI: {ROI_SIZE}x{ROI_SIZE}"
        #cv2.putText(roi_with_grid, info_text, (10, 30),
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        #cv2.putText(roi_with_grid, info_text, (10, 30),
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        
        # 7. 원본 이미지에 검출 박스 + ROI 표시
        img_with_box = img.copy()
        
        # 검출 박스 (빨간색)
        cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 0, 255), 3)
        
        # ROI 영역 (초록색)
        cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 중심점 표시
        cv2.circle(img_with_box, (center_x, center_y), 5, (255, 0, 0), -1)
        
        # 객체 번호 표시
        label = f"Object #{i+1}"
        cv2.putText(img_with_box, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 8. 결과 저장
        # 8-1. 원본 이미지 + 박스
        output_full = os.path.join(OUTPUT_FOLDER, f"object_{i+1:02d}_full_image.jpg")
        cv2.imwrite(output_full, img_with_box)
        print(f"  💾 저장: {output_full}")
        
        # 8-2. ROI만 (격자 없음)
        output_roi = os.path.join(OUTPUT_FOLDER, f"object_{i+1:02d}_roi.jpg")
        cv2.imwrite(output_roi, roi)
        print(f"  💾 저장: {output_roi}")
        
        # 8-3. ROI + 격자
        output_grid = os.path.join(OUTPUT_FOLDER, f"object_{i+1:02d}_roi_with_grid.jpg")
        cv2.imwrite(output_grid, roi_with_grid)
        print(f"  💾 저장: {output_grid}")
        
        # 9. (선택) 첫 번째 객체에 대해 셀별 히스토그램 시각화
        if i == 0:
            print(f"\n  🎨 격자 셀별 히스토그램 정보 생성 중...")
            visualize_cell_histograms(roi, roi_with_grid.copy(), GRID_SIZE, i+1)
    
    print(f"\n{'='*60}")
    print(f"✅ 작업 완료!")
    print(f"   출력 폴더: {OUTPUT_FOLDER}")
    print(f"{'='*60}")

def visualize_cell_histograms(roi, roi_grid, grid_size, obj_num):
    """
    각 셀의 히스토그램을 시각화 (선택적)
    발표자료용으로 몇 개 셀만 표시
    """
    h, w = roi.shape[:2]
    rows, cols = grid_size
    
    # HSV 변환
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 중앙 셀 (5, 5) 선택하여 히스토그램 표시
    center_r, center_c = rows // 2, cols // 2
    
    y_start = int((center_r / rows) * h)
    y_end = int(((center_r + 1) / rows) * h)
    x_start = int((center_c / cols) * w)
    x_end = int(((center_c + 1) / cols) * w)
    
    # 해당 셀 강조 표시
    roi_highlighted = roi_grid.copy()
    cv2.rectangle(roi_highlighted, (x_start, y_start), (x_end, y_end), (0, 0, 255), 3)
    
    # 텍스트 추가
    cell_text = f"Center Cell ({center_r},{center_c})"
    cv2.putText(roi_highlighted, cell_text, (x_start, y_start - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 저장
    output_highlighted = os.path.join(OUTPUT_FOLDER, f"object_{obj_num:02d}_roi_cell_highlighted.jpg")
    cv2.imwrite(output_highlighted, roi_highlighted)
    print(f"  💾 저장: {output_highlighted} (중앙 셀 강조)")
    
    # 셀 추출
    cell_hsv = hsv[y_start:y_end, x_start:x_end]
    cell_gray = gray[y_start:y_end, x_start:x_end]
    
    # 히스토그램 계산 (실제 사용하는 bin 수와 동일하게)
    mask = cv2.inRange(cell_hsv, (0, 0, 30), (180, 255, 255))
    hist_hsv = cv2.calcHist([cell_hsv], [0, 1], mask, [8, 4], [0, 180, 0, 256])
    hist_gray = cv2.calcHist([cell_gray], [0], mask, [16], [0, 256])
    
    print(f"     HSV 히스토그램 shape: {hist_hsv.shape} (Hue: 8 bins, Saturation: 4 bins)")
    print(f"     Gray 히스토그램 shape: {hist_gray.shape} (16 bins)")
    print(f"     총 특징 차원: {8*4 + 16} = 48차원 (셀당)")
    print(f"     전체 특징 차원: {48 * rows * cols} = {48 * rows * cols}차원 (11x11 격자)")

if __name__ == "__main__":
    main()
