import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# ==========================================
# 1. 유틸리티 함수
# ==========================================
def imread_unicode(filepath):
    try:
        stream = np.fromfile(filepath, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"오류: {e}")
        return None

def get_center_of_mass(channel_img):
    """
    단일 채널 이미지(Gray)에서 무게중심(Moment)을 계산하여 (x, y) 좌표 반환
    값이 없으면 None 반환
    """
    M = cv2.moments(channel_img)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    return None

def show_resized(window_name, img, width=800):
    if img is None: return
    h, w = img.shape[:2]
    if w > width:
        scale = width / w
        resized = cv2.resize(img, (int(w * scale), int(h * scale)))
        cv2.imshow(window_name, resized)
    else:
        cv2.imshow(window_name, img)

# ==========================================
# 2. 전역 변수
# ==========================================
diff_color = None
window_name = "ROI Center Tracking"

# ROI 크기 설정 (300x300)
ROI_SIZE = 300 

def on_trackbar(val):
    """
    슬라이더 조절 시 호출: Red 더하기 + ROI 무게중심 계산 + 그리기
    """
    global diff_color
    if diff_color is None: return

    # (1) 원본 차분 이미지 복사 (그리기 위해)
    display_img = diff_color.copy()
    h, w = display_img.shape[:2]

    # (2) Red 채널 값 더하기 (사용자 요청: Red 조절하면서 보기)
    b, g, r = cv2.split(display_img)
    r_added = cv2.add(r, val) # 슬라이더 값만큼 더함
    display_img = cv2.merge([b, g, r_added])

    # (3) 중앙 ROI 좌표 계산
    # 이미지 정중앙 좌표
    center_x, center_y = w // 2, h // 2
    
    # ROI 시작/끝 좌표 (300x300)
    half_roi = ROI_SIZE // 2
    x1 = max(0, center_x - half_roi)
    y1 = max(0, center_y - half_roi)
    x2 = min(w, center_x + half_roi)
    y2 = min(h, center_y + half_roi)

    # (4) ROI 추출 및 채널 분리
    roi = display_img[y1:y2, x1:x2]
    roi_b, roi_g, roi_r = cv2.split(roi)

    # (5) 무게중심 계산 (Green, Blue)
    # ROI 내부 좌표로 나옴 -> 나중에 전역 좌표(x1, y1)를 더해줘야 함
    center_g = get_center_of_mass(roi_g)
    center_b = get_center_of_mass(roi_b)

    # ================= 그리기 (Visualization) =================

    # 1. ROI 박스 그리기 (노란색)
    cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(display_img, "ROI (300x300)", (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 2. Green 무게중심 표시 (초록색 X)
    if center_g:
        gx, gy = center_g
        # ROI 내부 좌표 -> 전체 이미지 좌표 변환
        global_gx = x1 + gx
        global_gy = y1 + gy
        
        cv2.drawMarker(display_img, (global_gx, global_gy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(display_img, f"G:({global_gx},{global_gy})", (global_gx + 10, global_gy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 3. Blue 무게중심 표시 (파란색 X)
    if center_b:
        bx, by = center_b
        global_bx = x1 + bx
        global_by = y1 + by
        
        cv2.drawMarker(display_img, (global_bx, global_by), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 20, 2)
        cv2.putText(display_img, f"B:({global_bx},{global_by})", (global_bx + 10, global_by + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)

    # 4. 결과 출력
    show_resized(window_name, display_img)

# ==========================================
# 3. 메인 실행
# ==========================================
root = tk.Tk()
root.withdraw()
current_dir = os.getcwd()

print(">> 레이저 ON 사진 선택")
path_on = filedialog.askopenfilename(initialdir=current_dir, title="1. 레이저 ON")
if not path_on: exit()

print(">> 레이저 OFF 사진 선택")
path_off = filedialog.askopenfilename(initialdir=os.path.dirname(path_on), title="2. 레이저 OFF")
if not path_off: exit()

img_on = imread_unicode(path_on)
img_off = imread_unicode(path_off)

if img_on is None or img_off is None:
    print("이미지 로드 실패")
    exit()

# 차분 이미지 계산 (절댓값 차이)
diff_color = cv2.absdiff(img_on, img_off)

# 윈도우 생성
cv2.namedWindow(window_name)

# 트랙바 생성 (Red Add 조절용)
cv2.createTrackbar("Red Add", window_name, 0, 255, on_trackbar)

# 초기 실행
on_trackbar(0)

print("\n[사용법]")
print("- 노란색 박스: 이미지 중앙 300x300 ROI")
print("- 초록색 X: ROI 안에서 Green 성분의 무게중심")
print("- 파란색 X: ROI 안에서 Blue 성분의 무게중심")
print("- 슬라이더: Red 성분을 더해서 레이저 위치 확인용")

cv2.waitKey(0)
cv2.destroyAllWindows()