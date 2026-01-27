import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_laser_coordinates(image_path_1, image_path_2, roi_size=800):
    # --- 1. 이미지 읽기 헬퍼 함수 ---
    def imread_korean(path):
        try:
            with open(path, 'rb') as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            return None
    
    img1_full = imread_korean(image_path_1)
    img2_full = imread_korean(image_path_2)

    if img1_full is None or img2_full is None:
        print("❌ 이미지를 찾을 수 없습니다.")
        return

    if img1_full.shape != img2_full.shape:
        img2_full = cv2.resize(img2_full, (img1_full.shape[1], img1_full.shape[0]))

    # --- 2. ROI 설정 (중앙 자르기) ---
    h, w = img1_full.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    half_roi = roi_size // 2
    x1 = max(0, center_x - half_roi) # ROI 시작 X (Offset X)
    y1 = max(0, center_y - half_roi) # ROI 시작 Y (Offset Y)
    x2 = min(w, center_x + half_roi)
    y2 = min(h, center_y + half_roi)

    # 원본 보존을 위해 복사본 생성 (그림 그리기용)
    img_full_vis = img2_full.copy() # 전체 화면에 그릴 것
    
    # ROI 자르기
    img1_roi = img1_full[y1:y2, x1:x2]
    img2_roi = img2_full[y1:y2, x1:x2]
    
    # ROI 시각화용 복사본
    img_roi_vis = img2_roi.copy()

    # --- 3. 이미지 처리 (Diff & Threshold) ---
    # Com_test 방식: 블러 없이 Diff 계산
    diff_roi = cv2.absdiff(img2_roi, img1_roi)
    
    # Convert to grayscale
    gray = cv2.cvtColor(diff_roi, cv2.COLOR_BGR2GRAY)
    
    # Com_test 방식: THRESH_TOZERO with threshold 70
    cv_thresh = 70
    _, binary_diff_roi = cv2.threshold(gray, cv_thresh, 255, cv2.THRESH_TOZERO)

    # --- 4. 레이저 중심 좌표 찾기 (Moments) ---
    # Com_test 방식: Contour 없이 전체 이미지에서 moments 직접 계산
    M = cv2.moments(binary_diff_roi)
    
    laser_detected = False
    cx, cy = 0, 0 # ROI 내 좌표
    gx, gy = 0, 0 # 전체 좌표 (Global)

    # 무게 중심 계산 (분모가 0이 아닐 때만)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # 📍 좌표 변환: ROI 좌표 -> 전체 좌표
        gx = cx + x1
        gy = cy + y1
        
        laser_detected = True

    # --- 5. 시각화 (좌표 찍기) ---
    if laser_detected:
        # (A) ROI 이미지에 십자가 그리기 (빨간색)
        cv2.drawMarker(img_roi_vis, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.putText(img_roi_vis, f"ROI: {cx},{cy}", (cx + 10, cy - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # (B) 전체 이미지에 십자가 그리기 (빨간색) + ROI 박스 (초록색)
        cv2.rectangle(img_full_vis, (x1, y1), (x2, y2), (0, 255, 0), 5) # ROI 박스
        cv2.drawMarker(img_full_vis, (gx, gy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=50, thickness=5)
        cv2.putText(img_full_vis, f"Global: {gx},{gy}", (gx + 20, gy - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        print(f"✅ 레이저 검출 성공!")
        print(f"   - ROI 내 좌표: ({cx}, {cy})")
        print(f"   - 전체 좌표  : ({gx}, {gy})")
    else:
        print("⚠️ 레이저를 찾지 못했습니다.")

    # --- 6. Matplotlib 출력 ---
    plt.figure(figsize=(14, 10))

    # 1. ROI 결과 (Binary Mask)
    plt.subplot(2, 2, 1)
    plt.title("1. ROI Binary Mask (Processing)")
    plt.imshow(binary_diff_roi, cmap='gray')
    plt.axis('off')

    # 2. ROI 결과 (좌표 표시)
    plt.subplot(2, 2, 2)
    plt.title(f"2. Detected Laser in ROI ({cx}, {cy})")
    plt.imshow(cv2.cvtColor(img_roi_vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # 3. 전체 화면 결과 (좌표 표시)
    plt.subplot(2, 1, 2) # 아래쪽 전체 사용
    plt.title(f"3. Full Image with Coordinates ({gx}, {gy})")
    plt.imshow(cv2.cvtColor(img_full_vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- 실행 ---
show_laser_coordinates(
    r'c:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\추계 이후자료\레이저 HSV 확인용\captures_gui_20251126_203956\snap_20251126_204724_ud.jpg',
    r'c:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\추계 이후자료\레이저 HSV 확인용\captures_gui_20251126_203956\snap_20251126_204715_ud.jpg',
    roi_size=400
)