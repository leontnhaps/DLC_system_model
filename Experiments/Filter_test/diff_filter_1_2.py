#!/usr/bin/env python3
"""
LED 차분 이미지 유니버설 튜너 (거리 통합형)
- 모드 변경 없이 가까운 거리(Yellow)와 먼 거리(Red)를 동시 검출
- 파란색(Blue) 차단을 핵심으로 사용하는 로직
"""

import cv2
import numpy as np
from tkinter import Tk, filedialog
import os
import sys

def nothing(x):
    pass

def load_image_with_hangul(image_path):
    try:
        with open(image_path, 'rb') as f:
            image_array = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"❌ 이미지 로드 실패: {e}")
        return None

def select_two_images():
    print("\n[시스템] 파일 선택창을 엽니다...")
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    initial_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(">> 1. LED ON (빛 받은) 이미지를 선택하세요.")
    path_on = filedialog.askopenfilename(
        initialdir=initial_dir, title="1. LED ON 이미지 선택",
        filetypes=(("이미지 파일", "*.jpg *.jpeg *.png *.bmp"), ("모든 파일", "*.*")),
        parent=root
    )
    if not path_on: root.destroy(); return None, None
    
    print(">> 2. LED OFF (빛 없는) 이미지를 선택하세요.")
    path_off = filedialog.askopenfilename(
        initialdir=initial_dir, title="2. LED OFF 이미지 선택",
        filetypes=(("이미지 파일", "*.jpg *.jpeg *.png *.bmp"), ("모든 파일", "*.*")),
        parent=root
    )
    if not path_off: root.destroy(); return None, None

    root.destroy()
    return path_on, path_off

# ==========================================
# 1. 이미지 로드
# ==========================================
path_on, path_off = select_two_images()
if not path_on or not path_off: sys.exit()

img_on = load_image_with_hangul(path_on)
img_off = load_image_with_hangul(path_off)

if img_on is None or img_off is None: sys.exit()

# 차분 이미지 계산
diff_img_original = cv2.absdiff(img_on, img_off)

# 리사이징
height, width = diff_img_original.shape[:2]
scale_ratio = 800 / width
new_dim = (800, int(height * scale_ratio))
diff_img_resized = cv2.resize(diff_img_original, new_dim)

# ==========================================
# 2. GUI 설정
# ==========================================
cv2.namedWindow('Universal Color Tuner')

# 1. 밝기 증폭 (안보이는거 보이게)
cv2.createTrackbar('Boost (Norm)', 'Universal Color Tuner', 1, 1, nothing)

# 2. 최소 밝기 (노이즈 제거)
cv2.createTrackbar('Min Brightness', 'Universal Color Tuner', 30, 255, nothing)

# 3. [핵심 1] 하얀색(형광등) 차단 강도
# 빨간색이 파란색보다 얼마나 더 커야 하는가? (높을수록 엄격하게 하얀색 차단)
cv2.createTrackbar('White Cut (R>B)', 'Universal Color Tuner', 20, 100, nothing)

# 4. [핵심 2] 노란색 허용 범위 (Yellow Range)
# 이 값을 올리면 "완전 빨강" 뿐만 아니라 "노르스름한 빨강"도 인식함
# 0이면: G가 R보다 작아야 함 (Strict Red)
# 50이면: G가 R보다 50만큼 커도 봐줌 (Allow Yellow)
cv2.createTrackbar('Yellow Range', 'Universal Color Tuner', 50, 255, nothing)

print("\n-------------------------------------------")
print("📌 튜닝 팁:")
print("1. 'White Cut (R>B)': 배경의 하얀 조명이 사라질 때까지 올리세요.")
print("2. 'Yellow Range': 가까이 있는 타겟(노란색)이 잡힐 때까지 올리세요.")
print("-------------------------------------------\n")

while True:
    boost_on = cv2.getTrackbarPos('Boost (Norm)', 'Universal Color Tuner')
    min_bright = cv2.getTrackbarPos('Min Brightness', 'Universal Color Tuner')
    diff_b = cv2.getTrackbarPos('White Cut (R>B)', 'Universal Color Tuner')
    yellow_range = cv2.getTrackbarPos('Yellow Range', 'Universal Color Tuner')

    # 1. 전처리 (Boost)
    current_img = diff_img_resized.copy()
    if boost_on == 1:
        current_img = cv2.normalize(current_img, None, 0, 255, cv2.NORM_MINMAX)
        current_img = cv2.GaussianBlur(current_img, (3, 3), 0)
        cv2.putText(current_img, "BOOST ON", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 2. 채널 분리
    B, G, R = cv2.split(current_img)
    R_int = R.astype(np.int16)
    G_int = G.astype(np.int16)
    B_int = B.astype(np.int16)

    # 3. 유니버설 로직 적용
    
    # (A) 밝기 필터: 빨간색 채널이 일정 이상 밝아야 함
    mask_bright = (R > min_bright)

    # (B) 화이트 컷 (White Cut): R이 B보다 커야 함 (가장 중요)
    # R - B > diff_b
    mask_white_cut = (R_int - B_int) > diff_b

    # (C) 노란색 범위 (Yellow Range): R과 G의 관계
    # R - G > -yellow_range  (즉, G가 R + yellow_range 보다 작으면 통과)
    # 예: range가 50이면, R=200일 때 G가 250이어도 통과됨 (노란색 허용)
    mask_color_range = (R_int - G_int) > -yellow_range
    
    # 최종 마스크
    final_mask = mask_bright & mask_white_cut & mask_color_range
    final_mask = final_mask.astype(np.uint8) * 255

    # 4. 노이즈 제거 (Morphology)
    kernel = np.ones((3,3), np.uint8)
    final_mask = cv2.erode(final_mask, kernel, iterations=1)
    final_mask = cv2.dilate(final_mask, kernel, iterations=2)

    # 5. 시각화
    result_view = current_img.copy()
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        max_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_cnt)
        
        if area > 5:
            x, y, w, h = cv2.boundingRect(max_cnt)
            # 통합 모드라 박스 색상은 하나로 통일 (Cyan)
            cv2.rectangle(result_view, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cx, cy = x + w//2, y + h//2
            cv2.drawMarker(result_view, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            
            info = f"Center:({cx},{cy}) Area:{int(area)}"
            cv2.putText(result_view, info, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((result_view, mask_bgr))

    cv2.imshow('Universal Color Tuner', stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()