#!/usr/bin/env python3
"""
LED ON/OFF 차분 이미지 + 밝기 증폭(Normalize) + HSV 필터링 툴
"""

import cv2
import numpy as np
from tkinter import Tk, filedialog
import os

def nothing(x):
    pass

def load_image_with_hangul(image_path):
    with open(image_path, 'rb') as f:
        image_array = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return img

def select_image(title="이미지 선택"):
    root = Tk()
    root.withdraw()
    initial_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = filedialog.askopenfilename(
        initialdir=initial_dir, title=title,
        filetypes=(("이미지 파일", "*.jpg *.jpeg *.png *.bmp"), ("모든 파일", "*.*"))
    )
    root.destroy()
    return image_path

# ==========================================
# 1. 이미지 선택
# ==========================================
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("LED 차분 이미지 HSV + 부스터(Boost) 툴")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

print(">> LED ON 이미지를 선택하세요...")
path_on = select_image("1. LED ON 이미지 선택")
if not path_on: exit()

print(">> LED OFF 이미지를 선택하세요...")
path_off = select_image("2. LED OFF 이미지 선택")
if not path_off: exit()

# ==========================================
# 2. 이미지 로드 및 차분 계산
# ==========================================
img_on = load_image_with_hangul(path_on)
img_off = load_image_with_hangul(path_off)

if img_on is None or img_off is None:
    print("❌ 이미지를 읽을 수 없습니다.")
    exit()

# 차분 이미지 계산 (원본)
diff_img_original = cv2.absdiff(img_on, img_off)

# 리사이징
height, width = diff_img_original.shape[:2]
scale_ratio = 800 / width
new_dim = (800, int(height * scale_ratio))
diff_img_resized = cv2.resize(diff_img_original, new_dim)

print(f"✓ 이미지 준비 완료: {diff_img_resized.shape[1]}x{diff_img_resized.shape[0]}")

# ==========================================
# 3. GUI 설정
# ==========================================
cv2.namedWindow('HSV Booster Tuner')

# 트랙바 생성
cv2.createTrackbar('H Min', 'HSV Booster Tuner', 0, 179, nothing)
cv2.createTrackbar('H Max', 'HSV Booster Tuner', 179, 179, nothing)
cv2.createTrackbar('S Min', 'HSV Booster Tuner', 0, 255, nothing)
cv2.createTrackbar('S Max', 'HSV Booster Tuner', 255, 255, nothing)
cv2.createTrackbar('V Min', 'HSV Booster Tuner', 50, 255, nothing)
cv2.createTrackbar('V Max', 'HSV Booster Tuner', 255, 255, nothing)

# [핵심] 부스터 기능 추가 (0: OFF, 1: ON)
cv2.createTrackbar('Boost (Norm)', 'HSV Booster Tuner', 1, 1, nothing) 
cv2.createTrackbar('Threshold', 'HSV Booster Tuner', 30, 255, nothing)

print("\nTip: 'Boost (Norm)'을 1로 켜면 어두운 부분이 밝게 증폭됩니다!")

while True:
    # 1. 트랙바 값 읽기
    h_min = cv2.getTrackbarPos('H Min', 'HSV Booster Tuner')
    h_max = cv2.getTrackbarPos('H Max', 'HSV Booster Tuner')
    s_min = cv2.getTrackbarPos('S Min', 'HSV Booster Tuner')
    s_max = cv2.getTrackbarPos('S Max', 'HSV Booster Tuner')
    v_min = cv2.getTrackbarPos('V Min', 'HSV Booster Tuner')
    v_max = cv2.getTrackbarPos('V Max', 'HSV Booster Tuner')
    
    boost_on = cv2.getTrackbarPos('Boost (Norm)', 'HSV Booster Tuner')
    threshold = cv2.getTrackbarPos('Threshold', 'HSV Booster Tuner')

    # 2. 이미지 전처리 (부스트 적용 여부)
    current_img = diff_img_resized.copy()

    if boost_on == 1:
        # [핵심 로직] 정규화(밝기 증폭) + 블러
        current_img = cv2.normalize(current_img, None, 0, 255, cv2.NORM_MINMAX)
        current_img = cv2.GaussianBlur(current_img, (5, 5), 0)
        
        # 부스트 켜졌을 때 텍스트 표시
        cv2.putText(current_img, "BOOST ON", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 3. 그레이스케일 Threshold (노이즈 제거용)
    gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
    _, thresh_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    img_filtered = cv2.bitwise_and(current_img, current_img, mask=thresh_mask)

    # 4. HSV 변환 및 마스킹
    hsv = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    hsv_mask = cv2.inRange(hsv, lower, upper)

    # 모폴로지 (노이즈 제거)
    kernel = np.ones((3,3), np.uint8)
    hsv_mask = cv2.erode(hsv_mask, kernel, iterations=1)
    hsv_mask = cv2.dilate(hsv_mask, kernel, iterations=2)

    # 5. 결과 시각화 (컨투어 그리기)
    result_view = current_img.copy()
    contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 가장 큰 물체 찾기
    if contours:
        max_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_cnt) > 5: # 최소 크기 필터
            x, y, w, h = cv2.boundingRect(max_cnt)
            # 초록색 박스
            cv2.rectangle(result_view, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # 빨간색 컨투어
            cv2.drawContours(result_view, [max_cnt], -1, (0, 0, 255), 2)
            # 좌표 표시
            center_text = f"X:{x+w//2} Y:{y+h//2}"
            cv2.putText(result_view, center_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 화면 병합 (왼쪽: 처리된 이미지 / 오른쪽: 마스크)
    mask_bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((result_view, mask_bgr))

    cv2.imshow('HSV Booster Tuner', stacked)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

cv2.destroyAllWindows()