#!/usr/bin/env python3
"""
LED 차분 이미지 RGB 채널 필터링 (Red Dominance) 튜너 - 안정화 버전
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
    """Tkinter 인스턴스를 한 번만 생성하여 연속으로 파일 선택 (안정성 확보)"""
    print("\n[시스템] 파일 선택창을 엽니다...")
    
    root = Tk()
    root.withdraw() # 빈 창 숨기기
    root.attributes('-topmost', True) # 창을 맨 앞으로 가져오기 (가려짐 방지)

    initial_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. ON 이미지
    print(">> 1. LED ON 이미지를 선택하세요.")
    path_on = filedialog.askopenfilename(
        initialdir=initial_dir, title="1. LED ON 이미지 선택",
        filetypes=(("이미지 파일", "*.jpg *.jpeg *.png *.bmp"), ("모든 파일", "*.*")),
        parent=root
    )
    
    if not path_on:
        print("❌ LED ON 선택이 취소되었습니다.")
        root.destroy()
        return None, None

    print(f"   선택됨: {os.path.basename(path_on)}")
    
    # 2. OFF 이미지
    print(">> 2. LED OFF 이미지를 선택하세요.")
    path_off = filedialog.askopenfilename(
        initialdir=initial_dir, title="2. LED OFF 이미지 선택",
        filetypes=(("이미지 파일", "*.jpg *.jpeg *.png *.bmp"), ("모든 파일", "*.*")),
        parent=root
    )

    if not path_off:
        print("❌ LED OFF 선택이 취소되었습니다.")
        root.destroy()
        return None, None
        
    print(f"   선택됨: {os.path.basename(path_off)}")

    root.destroy() # 모든 선택이 끝나면 안전하게 종료
    return path_on, path_off

# ==========================================
# 1. 이미지 로드
# ==========================================
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("RGB 채널 필터 (Red Dominance) 튜너")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# 여기서 한 번에 두 개 다 고릅니다
path_on, path_off = select_two_images()

if not path_on or not path_off:
    print("프로그램을 종료합니다.")
    sys.exit()

img_on = load_image_with_hangul(path_on)
img_off = load_image_with_hangul(path_off)

if img_on is None or img_off is None:
    print("❌ 이미지를 읽을 수 없습니다.")
    sys.exit()

# 차분 이미지 계산
print("[시스템] 차분 이미지 계산 중...")
diff_img_original = cv2.absdiff(img_on, img_off)

# 리사이징
height, width = diff_img_original.shape[:2]
scale_ratio = 800 / width
new_dim = (800, int(height * scale_ratio))
diff_img_resized = cv2.resize(diff_img_original, new_dim)

print(f"✓ 준비 완료: {diff_img_resized.shape[1]}x{diff_img_resized.shape[0]}")
print("[시스템] 튜너 창을 띄웁니다. (안 보이면 작업표시줄 확인)")

# ==========================================
# 2. GUI 설정
# ==========================================
cv2.namedWindow('RGB Filter Tuner')

# 트랙바 생성
cv2.createTrackbar('Boost (Norm)', 'RGB Filter Tuner', 1, 1, nothing)
cv2.createTrackbar('Min Red (Abs)', 'RGB Filter Tuner', 20, 255, nothing)
cv2.createTrackbar('Red > Green', 'RGB Filter Tuner', 10, 100, nothing)
cv2.createTrackbar('Red > Blue', 'RGB Filter Tuner', 10, 100, nothing)

while True:
    # 값 읽기
    boost_on = cv2.getTrackbarPos('Boost (Norm)', 'RGB Filter Tuner')
    min_r = cv2.getTrackbarPos('Min Red (Abs)', 'RGB Filter Tuner')
    diff_g = cv2.getTrackbarPos('Red > Green', 'RGB Filter Tuner')
    diff_b = cv2.getTrackbarPos('Red > Blue', 'RGB Filter Tuner')

    # 1. 전처리 (Boost)
    current_img = diff_img_resized.copy()
    if boost_on == 1:
        current_img = cv2.normalize(current_img, None, 0, 255, cv2.NORM_MINMAX)
        current_img = cv2.GaussianBlur(current_img, (3, 3), 0)
        cv2.putText(current_img, "BOOST ON", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 2. 채널 분리
    B, G, R = cv2.split(current_img)

    # 3. 로직 적용
    mask_abs = (R > min_r)
    
    # int16으로 변환하여 뺄셈 시 언더플로우 방지
    R_int = R.astype(np.int16)
    G_int = G.astype(np.int16)
    B_int = B.astype(np.int16)

    mask_rg = (R_int - G_int) > diff_g
    mask_rb = (R_int - B_int) > diff_b
    
    final_mask = mask_abs & mask_rg & mask_rb
    final_mask = final_mask.astype(np.uint8) * 255

    # 4. 노이즈 제거
    kernel = np.ones((3,3), np.uint8)
    final_mask = cv2.erode(final_mask, kernel, iterations=1)
    final_mask = cv2.dilate(final_mask, kernel, iterations=2)

    # 5. 시각화
    result_view = current_img.copy()
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        max_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_cnt)
        if area > 2:
            x, y, w, h = cv2.boundingRect(max_cnt)
            cv2.rectangle(result_view, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cx, cy = x + w//2, y + h//2
            cv2.drawMarker(result_view, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            info = f"Center:({cx},{cy}) Area:{int(area)}"
            cv2.putText(result_view, info, (10, result_view.shape[0]-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((result_view, mask_bgr))

    cv2.imshow('RGB Filter Tuner', stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()