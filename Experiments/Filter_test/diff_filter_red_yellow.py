#!/usr/bin/env python3
"""
LED 차분 이미지 스마트 컬러 튜너 (Red & Yellow 통합)
- 거리에 따른 색상 변화(Red <-> Yellow) 모두 대응 가능
- "BOOST" 글자 인식 버그 수정됨
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
cv2.namedWindow('Smart Color Tuner')

# 트랙바 생성
cv2.createTrackbar('Boost (Norm)', 'Smart Color Tuner', 1, 1, nothing)
cv2.createTrackbar('Min Red (Abs)', 'Smart Color Tuner', 30, 255, nothing)

# [핵심] Yellow Mode: R > G 조건을 무시하거나 완화함
cv2.createTrackbar('Include Yellow', 'Smart Color Tuner', 1, 1, nothing) 

# R > G: 노란색 포함 모드일 때는 이 값이 무시되거나 낮게 적용됨
cv2.createTrackbar('Red > Green', 'Smart Color Tuner', 10, 100, nothing)

# R > B: 하얀색 빛(형광등 등)을 걸러내는 핵심 필터 (항상 중요)
cv2.createTrackbar('Red > Blue', 'Smart Color Tuner', 20, 100, nothing)

print("\n[Tip] 가까운 거리(노란색) 인식이 필요하면 'Include Yellow'를 1로 켜세요!")

while True:
    boost_on = cv2.getTrackbarPos('Boost (Norm)', 'Smart Color Tuner')
    min_r = cv2.getTrackbarPos('Min Red (Abs)', 'Smart Color Tuner')
    include_yellow = cv2.getTrackbarPos('Include Yellow', 'Smart Color Tuner')
    diff_g = cv2.getTrackbarPos('Red > Green', 'Smart Color Tuner')
    diff_b = cv2.getTrackbarPos('Red > Blue', 'Smart Color Tuner')

    # 1. 전처리 (Boost)
    current_img = diff_img_resized.copy()
    if boost_on == 1:
        current_img = cv2.normalize(current_img, None, 0, 255, cv2.NORM_MINMAX)
        current_img = cv2.GaussianBlur(current_img, (3, 3), 0)

    # 2. 채널 분리
    B, G, R = cv2.split(current_img)
    R_int = R.astype(np.int16)
    G_int = G.astype(np.int16)
    B_int = B.astype(np.int16)

    # 3. 로직 적용
    # (A) 기본 밝기 필터
    mask_abs = (R > min_r)

    # (B) 하얀색 차단 필터 (가장 중요)
    # R과 B의 차이가 커야 함 (하얀색은 R≒B 이므로 걸러짐)
    mask_rb = (R_int - B_int) > diff_b

    # (C) 노란색/빨간색 구분 필터
    if include_yellow == 1:
        # 노란색 허용 모드: R이 G보다 작아도 됨 (단, 너무 초록색이면 안 되니 약한 조건만)
        # 예: R이 G보다 30 이상 작지만 않으면 통과 (G가 R보다 조금 커도 됨)
        mask_rg = (R_int - G_int) > -30 
    else:
        # 오직 빨간색만: R이 G보다 확실히 커야 함
        mask_rg = (R_int - G_int) > diff_g
    
    final_mask = mask_abs & mask_rb & mask_rg
    final_mask = final_mask.astype(np.uint8) * 255

    # 4. 노이즈 제거
    kernel = np.ones((3,3), np.uint8)
    final_mask = cv2.erode(final_mask, kernel, iterations=1)
    final_mask = cv2.dilate(final_mask, kernel, iterations=2)

    # 5. 시각화 (좌표 표시)
    result_view = current_img.copy()
    if boost_on == 1:
        cv2.putText(result_view, "BOOST ON", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if include_yellow == 1:
        cv2.putText(result_view, "YELLOW MODE", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 가장 큰 영역 찾기
        max_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_cnt)
        
        # 노이즈 크기 필터
        if area > 5:
            x, y, w, h = cv2.boundingRect(max_cnt)
            # 노란색 모드면 노란 박스, 아니면 빨간 박스
            color = (0, 255, 255) if include_yellow else (0, 0, 255)
            
            cv2.rectangle(result_view, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cx, cy = x + w//2, y + h//2
            cv2.drawMarker(result_view, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)
            
            info = f"Center:({cx},{cy}) Area:{int(area)}"
            cv2.putText(result_view, info, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((result_view, mask_bgr))

    cv2.imshow('Smart Color Tuner', stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()