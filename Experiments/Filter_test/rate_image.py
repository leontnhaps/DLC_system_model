#!/usr/bin/env python3
"""
LED 변화율(Ratio) 이미지 뷰어
- 차분(빼기) 대신 비율(나누기)을 사용하여 밝은 배경 노이즈를 제거
- 공식: (ON 이미지) / (OFF 이미지)
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

def main():
    while True:
        path_on, path_off = select_two_images()
        if not path_on or not path_off:
            print("프로그램을 종료합니다.")
            break

        img_on = load_image_with_hangul(path_on)
        img_off = load_image_with_hangul(path_off)

        if img_on is None or img_off is None: continue

        # 1. 실수형(float)으로 변환 (나눗셈을 위해)
        # 0으로 나누는 것을 방지하기 위해 +1을 더함
        img_on_f = img_on.astype(np.float32) + 1.0
        img_off_f = img_off.astype(np.float32) + 1.0

        # 2. 비율 계산 (ON / OFF)
        # 값이 클수록 '몇 배' 더 밝아졌는지를 의미
        ratio_map = img_on_f / img_off_f

        # 3. 채널 합치기 (평균 변화율)
        # R, G, B 채널 중 가장 변화가 큰 값을 사용하거나 평균을 사용
        # 여기서는 최대 변화율을 사용 (가장 민감하게)
        ratio_gray = np.max(ratio_map, axis=2)

        # 리사이징
        height, width = ratio_gray.shape[:2]
        scale_ratio = 800 / width
        new_dim = (800, int(height * scale_ratio))
        ratio_resized = cv2.resize(ratio_gray, new_dim)

        cv2.namedWindow('Ratio Filter')
        
        # 슬라이더: 최소 배율 (예: 15 -> 1.5배)
        cv2.createTrackbar('Min Ratio (x10)', 'Ratio Filter', 12, 100, nothing)
        
        # 슬라이더: 시각화 증폭 (화면이 너무 어두우면 올리세요)
        cv2.createTrackbar('View Gain', 'Ratio Filter', 50, 255, nothing)

        print("\n-------------------------------------------")
        print("➗ 변화율(Ratio) 모드")
        print("- Min Ratio: 이 값보다 적게 변한 건 다 무시합니다.")
        print("  (예: 20으로 설정하면 2배 이상 밝아진 것만 표시)")
        print("-------------------------------------------\n")

        while True:
            # 슬라이더 값 읽기 (10을 나누어서 실수로 사용)
            th_val = cv2.getTrackbarPos('Min Ratio (x10)', 'Ratio Filter')
            threshold = th_val / 10.0  # 예: 15 -> 1.5배

            gain = cv2.getTrackbarPos('View Gain', 'Ratio Filter')

            # --- [핵심 로직] ---
            # 변화율이 threshold보다 큰 픽셀만 남김
            mask = (ratio_resized > threshold).astype(np.uint8) * 255

            # 노이즈 제거
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # 시각화용 이미지 생성
            # 비율 맵을 눈에 보이게 0~255로 변환 (배경 노이즈랑 구분 쉽게)
            # (ratio - 1.0) * Gain
            display_ratio = (ratio_resized - 1.0) * gain
            display_ratio = np.clip(display_ratio, 0, 255).astype(np.uint8)
            
            # 마스크 적용
            result = cv2.bitwise_and(display_ratio, display_ratio, mask=mask)
            
            # 컬러 변환 (보기 좋게)
            result_color = cv2.applyColorMap(result, cv2.COLORMAP_JET)
            
            # 마스크가 없는 부분(배경)은 검은색 처리
            result_color[mask == 0] = 0

            cv2.imshow('Ratio Filter', result_color)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                return 
            elif key == ord('r') or key == ord('R'):
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()