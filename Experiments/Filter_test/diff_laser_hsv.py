#!/usr/bin/env python3
"""
Single Image Threshold Tuner
- 기능: 이미지 1장 로드 -> Gray 변환 -> Threshold 조절 -> 확인
"""

import cv2
import numpy as np
from tkinter import Tk, filedialog
import os
import sys

def nothing(x):
    pass

# 한글 경로 이미지 로드용 함수
def load_image_with_hangul(image_path):
    try:
        with open(image_path, 'rb') as f:
            image_array = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"❌ 이미지 로드 실패: {e}")
        return None

# 파일 1개 선택
def select_single_image():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    initial_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(">> 분석할 이미지(Diff 결과물)를 선택하세요.")
    path = filedialog.askopenfilename(
        initialdir=initial_dir, title="이미지 선택",
        filetypes=(("이미지 파일", "*.jpg *.jpeg *.png *.bmp"), ("모든 파일", "*.*")),
        parent=root
    )
    root.destroy()
    return path

# ==========================================
# 메인 로직
# ==========================================
image_path = select_single_image()
if not image_path: sys.exit()

# 1. 이미지 로드
img_bgr = load_image_with_hangul(image_path)
if img_bgr is None: sys.exit()

# 2. 보기 좋게 리사이징 (기능엔 영향 없음)
height, width = img_bgr.shape[:2]
if width > 800:
    scale_ratio = 800 / width
    new_dim = (800, int(height * scale_ratio))
    img_resized = cv2.resize(img_bgr, new_dim)
else:
    img_resized = img_bgr

# 3. Gray Scale 변환
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# 윈도우 설정
window_name = 'Threshold Only'
cv2.namedWindow(window_name)
cv2.createTrackbar('Threshold', window_name, 50, 255, nothing)

print(f"\n[로드 완료] {os.path.basename(image_path)}")
print(">> 슬라이더를 움직여서 원하는 밝기 기준(Threshold)을 찾으세요.")

while True:
    # 값 읽기
    th = cv2.getTrackbarPos('Threshold', window_name)

    # 4. Threshold 적용
    _, binary = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)

    # 시각화 (왼쪽: Gray, 오른쪽: Binary)
    gray_view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    binary_view = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    # 텍스트
    cv2.putText(gray_view, "Input (Gray)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(binary_view, f"Result (Th={th})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    stacked = np.hstack((gray_view, binary_view))
    cv2.imshow(window_name, stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()