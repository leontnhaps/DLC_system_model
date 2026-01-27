#!/usr/bin/env python3
"""
LED 차분 이미지 뷰어 + RGB 채널별 분석기
- 기존 기능: 빨간색 필터링 결과 확인
- 추가 기능: R, G, B 채널을 각각 분리해서 눈으로 확인 (어떤 색 성분이 강한지 분석용)
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

        # 4. 리사이징 (보기 좋게)
        height, width = diff.shape[:2]
        scale_ratio = 800 / width
        new_dim = (800, int(height * scale_ratio))
        diff_resized = cv2.resize(diff, new_dim)

        # 5. 윈도우 생성
        window_name = 'RGB Channel Analyzer'
        cv2.namedWindow(window_name)
        
        # 슬라이더: 빨간색 최소 밝기
        cv2.createTrackbar('Red Threshold', window_name, 30, 255, nothing)

        print("\n-------------------------------------------")
        print("📊 RGB 채널 분석 모드")
        print("- 위쪽: 원본 차분 / 필터링 결과")
        print("- 아래쪽: R, G, B 채널별 밝기 (하얀색일수록 해당 색상이 강함)")
        print("-------------------------------------------\n")

        while True:
            # 슬라이더 값 읽기
            th = cv2.getTrackbarPos('Red Threshold', window_name)

            # --- [핵심] 채널 분리 ---
            B, G, R = cv2.split(diff_resized)

            # --- [로직] 단순 빨간색 필터 ---
            # 조건 1: 빨간색이 일정 밝기 이상일 것 (th)
            mask_bright = (R > th)
            # 조건 2: 빨간색이 초록색, 파란색보다 클 것
            mask_color = (R > G) & (R > B)

            # 최종 마스크
            mask = mask_bright & mask_color
            mask = mask.astype(np.uint8) * 255

            # 노이즈 제거
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)

            # 필터링된 이미지 만들기
            res = cv2.bitwise_and(diff_resized, diff_resized, mask=mask)

            # === [시각화] ===
            
            # 1. 상단: 원본 + 결과 (가로 1600px)
            row_top = np.hstack((diff_resized, res))
            
            # 2. 하단: R, G, B 채널 보여주기 (가로 1600px에 맞춰서 3등분)
            # 각각을 3채널(컬러) 이미지로 변환해야 hstack 가능
            R_view = cv2.cvtColor(R, cv2.COLOR_GRAY2BGR)
            G_view = cv2.cvtColor(G, cv2.COLOR_GRAY2BGR)
            B_view = cv2.cvtColor(B, cv2.COLOR_GRAY2BGR)

            # 라벨 표시
            cv2.putText(R_view, "RED Channel (Target)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(G_view, "GREEN Channel (Yellow)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(B_view, "BLUE Channel (Noise)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 하단 3개 이미지를 상단 너비(1600px)에 맞게 리사이징
            # 1600 / 3 = 약 533px
            total_width = row_top.shape[1]
            sub_width = total_width // 3
            sub_height = int(height * (sub_width / width)) # 비율 유지
            
            R_view = cv2.resize(R_view, (sub_width, sub_height))
            G_view = cv2.resize(G_view, (sub_width, sub_height))
            B_view = cv2.resize(B_view, (total_width - 2*sub_width, sub_height)) # 남은 공간 채우기

            row_bottom = np.hstack((R_view, G_view, B_view))

            # 상단, 하단 합치기
            # vstack을 위해 width가 같아야 하는데 계산 오차로 1~2픽셀 다를 수 있음 -> 리사이징으로 맞춤
            if row_top.shape[1] != row_bottom.shape[1]:
                 row_bottom = cv2.resize(row_bottom, (row_top.shape[1], row_bottom.shape[0]))

            final_view = np.vstack((row_top, row_bottom))

            # 안내 문구 추가
            cv2.putText(final_view, "Original Diff", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(final_view, f"Filtered Result (Th={th})", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow(window_name, final_view)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                return 
            elif key == ord('r') or key == ord('R'):
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()