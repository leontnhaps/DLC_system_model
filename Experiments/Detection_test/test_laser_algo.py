
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

def imread_unicode(filepath):
    """한글 경로를 포함한 파일을 읽기 위한 함수"""
    try:
        # numpy로 바이너리 읽기
        stream = np.fromfile(filepath, dtype=np.uint8)
        # OpenCV로 디코딩
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return None

def show_resized(window_name, img, scale=0.4):
    """화면이 너무 클 수 있어서 보기 좋게 줄여서 보여주는 함수"""
    if img is None: return
    h, w = img.shape[:2]
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(window_name, resized)

def find_laser_center(img_on, img_off):
    """
    Com/pointing_handler.py의 _find_laser_center 로직을 그대로 구현
    """
    if img_on is None or img_off is None:
        return None, None, None

    H, W = img_on.shape[:2]
    
    # 1. 관심 영역(ROI) 설정 (중앙 1400x1400 Crop)
    # pointing_handler.py:560
    crop_size = 1400  # 700 * 2
    cy, cx = H // 2, W // 2
    
    # Y축은 약간 위쪽을 ROI로 잡음 (레이저가 보통 위/아래로 튀므로)
    roi_y1 = max(0, cy - 700 - 100) 
    roi_y2 = min(H, cy + 700 - 100)
    roi_x1 = max(0, cx - 700)
    roi_x2 = min(W, cx + 700)
    
    roi_on = img_on[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_off = img_off[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # 2. Diff 계산
    diff = cv2.absdiff(roi_on, roi_off)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # grayscale
    
    # 3. Threshold (diff > 50인 픽셀만)
    # pointing_handler.py:587 (Threshold 50)
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # (Optional) Masking - 여기서는 테스트용이라 생략하거나 빈 리스트 처리
    # exclude_bboxes logic won't be applied here as we don't have YOLO input.
    
    # 4. 무게중심 계산
    ys, xs = np.where(mask > 0)
    
    # 시각화용 이미지 (ROI 위에 그림)
    vis_roi = roi_on.copy()
    
    # 마스크 영역 표시 (노란색)
    vis_roi[mask > 0] = (0, 255, 255)
    
    if len(xs) == 0:
        return None, vis_roi, mask
        
    # ROI 내부 좌표
    roi_cx = int(np.mean(xs))
    roi_cy = int(np.mean(ys))
    
    # 전체 이미지 좌표로 변환
    global_cx = roi_cx + roi_x1
    global_cy = roi_cy + roi_y1
    
    # 결과 시각화 (ROI 기준)
    cv2.circle(vis_roi, (roi_cx, roi_cy), 10, (0, 0, 255), 2)  # 빨간 원
    cv2.drawMarker(vis_roi, (roi_cx, roi_cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    
    return (global_cx, global_cy), vis_roi, mask


# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    print("파일 선택창을 띄웁니다...")
    
    # tkinter 기본 윈도우 숨기기
    root = tk.Tk()
    root.withdraw()
    
    # 현재 작업 경로
    current_dir = os.getcwd()
    
    # 1) 레이저 ON 사진 선택
    print(">> 레이저가 [켜진](ON) 사진을 선택해주세요.")
    path_on = filedialog.askopenfilename(
        title="1. 레이저 ON 사진 선택",
        initialdir=current_dir,
        filetypes=[("Image files", "*.jpg *.png *.jpeg")]
    )
    if not path_on:
        print("❌ 파일 선택 취소")
        exit()
        
    # 2) 레이저 OFF 사진 선택
    print(">> 레이저가 [꺼진](OFF) 사진을 선택해주세요.")
    path_off = filedialog.askopenfilename(
        title="2. 레이저 OFF 사진 선택",
        initialdir=os.path.dirname(path_on),
        filetypes=[("Image files", "*.jpg *.png *.jpeg")]
    )
    if not path_off:
        print("❌ 파일 선택 취소")
        exit()
        
    print(f"선택된 파일:\n ON: {os.path.basename(path_on)}\n OFF: {os.path.basename(path_off)}")
    
    # 이미지 로드
    img_on = imread_unicode(path_on)
    img_off = imread_unicode(path_off)
    
    if img_on is None or img_off is None:
        print("이미지 로드 실패")
        exit()
        
    # 알고리즘 실행
    center, vis_roi, mask = find_laser_center(img_on, img_off)
    
    if center:
        print(f"\n✅ 레이저 중심 발견: ({center[0]}, {center[1]})")
    else:
        print("\n❌ 레이저 중심 발견 실패 (Threshold 50 미달)")
        
    # 결과 보여주기
    print("\n결과 창 확인 후 아무 키나 누르면 종료됩니다.")
    show_resized("1. Laser ON (Original)", img_on)
    show_resized("2. ROI Result (Mask+Center)", vis_roi)
    show_resized("3. Binary Mask (Threshold 50)", mask)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
