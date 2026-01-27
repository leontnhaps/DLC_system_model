#!/usr/bin/env python3
"""
LED 차분 이미지 뷰어 + 빨간색 & 노란색 통합 필터
- 'Yellow Range'를 올리면 노란색(가까운 거리)도 잡힙니다.
- 'Blue Cut'으로 하얀색 조명을 제거합니다.
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
# 메인 실행부
# ==========================================
def main():
    while True:
        # 1. 이미지 선택
        path_on, path_off = select_two_images()
        if not path_on or not path_off:
            print("프로그램을 종료합니다.")
            break

        # 2. 이미지 로드
        img_on = load_image_with_hangul(path_on)
        img_off = load_image_with_hangul(path_off)

        if img_on is None or img_off is None:
            continue

        # 3. 차분 계산
        diff = cv2.absdiff(img_on, img_off)

        # 4. 리사이징
        height, width = diff.shape[:2]
        scale_ratio = 800 / width
        new_dim = (800, int(height * scale_ratio))
        diff_resized = cv2.resize(diff, new_dim)

        # 5. 필터링 윈도우 생성
        cv2.namedWindow('Red & Yellow Filter')
        
        # [슬라이더 1] 최소 밝기 (기본값 30)
        cv2.createTrackbar('Min Bright', 'Red & Yellow Filter', 30, 255, nothing)

        # [슬라이더 2] 노란색 허용 범위 (기본값 60)
        # 이 값을 올리면 "초록색이 섞인 빨강(노랑)"도 통과됩니다.
        cv2.createTrackbar('Yellow Range', 'Red & Yellow Filter', 60, 200, nothing)

        # [슬라이더 3] 파란색 차단 강도 (기본값 20)
        # 이 값을 올리면 하얀색 조명(형광등)이 사라집니다.
        cv2.createTrackbar('Blue Cut', 'Red & Yellow Filter', 20, 100, nothing)

        print("\n-------------------------------------------")
        print("🟡🔴 빨간색 + 노란색 통합 필터")
        print("1. 'Yellow Range'를 올려서 노란색 필름을 잡으세요.")
        print("2. 흰색 빛이 보이면 'Blue Cut'을 올리세요.")
        print("-------------------------------------------\n")

        while True:
            # 슬라이더 값 읽기
            min_bright = cv2.getTrackbarPos('Min Bright', 'Red & Yellow Filter')
            yellow_range = cv2.getTrackbarPos('Yellow Range', 'Red & Yellow Filter')
            blue_cut = cv2.getTrackbarPos('Blue Cut', 'Red & Yellow Filter')

            # --- [핵심 로직] ---
            B, G, R = cv2.split(diff_resized)
            
            # 계산을 위해 int16으로 변환 (음수 처리)
            R_int = R.astype(np.int16)
            G_int = G.astype(np.int16)
            B_int = B.astype(np.int16)

            # 조건 1: 밝기 필터 (빨간색이 일정 이상 밝아야 함)
            mask_bright = (R > min_bright)

            # 조건 2: 노란색 허용 (R과 G의 차이)
            # R - G > -yellow_range  =>  G가 R + yellow_range 보다 작으면 됨
            mask_yellow = (R_int - G_int) > -yellow_range

            # 조건 3: 하얀색 차단 (R과 B의 차이)
            # 빨간색이 파란색보다 확실히 커야 함 (하얀색은 R≒B 이므로 걸러짐)
            mask_blue_cut = (R_int - B_int) > blue_cut

            # 최종 마스크
            mask = mask_bright & mask_yellow & mask_blue_cut
            mask = mask.astype(np.uint8) * 255

            # 노이즈 제거
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)

            # 필터링된 이미지
            res = cv2.bitwise_and(diff_resized, diff_resized, mask=mask)

            # 화면 병합
            stacked = np.hstack((diff_resized, res))
            
            cv2.putText(stacked, "Original Diff", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(stacked, "Red+Yellow Filtered", (diff_resized.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow('Red & Yellow Filter', stacked)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                return 
            elif key == ord('r') or key == ord('R'):
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()