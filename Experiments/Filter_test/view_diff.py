import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

def imread_unicode(filepath):
    """한글 경로를 포함한 파일을 읽기 위한 함수"""
    try:
        # numpy로 바이너리 읽기
        stream = np.fromfile(filepath, dtype=np.uint8)
        # OpenCV로 디코딩
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return None

def show_resized(window_name, img, scale=0.4):
    """화면이 너무 클 수 있어서 보기 좋게 줄여서 보여주는 함수"""
    if img is None: return
    h, w = img.shape[:2]
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(window_name, resized)

# ==========================================
# 1. 파일 선택 (File Dialog)
# ==========================================
print("파일 선택창을 띄웁니다...")

# tkinter 기본 윈도우 숨기기 (이거 안 하면 빈 창이 하나 더 뜹니다)
root = tk.Tk()
root.withdraw()

# 현재 작업 경로 가져오기 (파일창이 여기서부터 열림)
current_dir = os.getcwd()

# 1) 레이저 ON 사진 선택
print(">> 레이저가 [켜진](ON) 사진을 선택해주세요.")
path_on = filedialog.askopenfilename(
    title="1. 레이저 ON 사진 선택",
    initialdir=current_dir,
    filetypes=[("Image files", "*.jpg *.png *.jpeg")]
)

# 취소 눌렀을 때 처리
if not path_on:
    print("❌ 파일 선택이 취소되었습니다.")
    exit()

# 2) 레이저 OFF 사진 선택
print(">> 레이저가 [꺼진](OFF) 사진을 선택해주세요.")
path_off = filedialog.askopenfilename(
    title="2. 레이저 OFF 사진 선택",
    initialdir=os.path.dirname(path_on), # 방금 선택한 폴더에서 다시 열기
    filetypes=[("Image files", "*.jpg *.png *.jpeg")]
)

if not path_off:
    print("❌ 파일 선택이 취소되었습니다.")
    exit()

print(f"선택된 파일:\n ON: {os.path.basename(path_on)}\n OFF: {os.path.basename(path_off)}")

# ==========================================
# 2. 이미지 처리 (차분 및 결과 확인)
# ==========================================
# 한글 경로 지원을 위해 imread_unicode 함수 사용
img_on = imread_unicode(path_on)
img_off = imread_unicode(path_off)

if img_on is None or img_off is None:
    print("이미지를 불러올 수 없습니다. 파일이 손상되었거나 이미지 파일이 아닐 수 있습니다.")
    exit()

# ==========================================
# A. 컬러 차분 (Color Difference)
# ==========================================
# 흑백 변환 없이 바로 차분 계산
color_diff = cv2.absdiff(img_on, img_off)

# 컬러 차분을 그레이스케일로 변환하여 threshold 적용
color_diff_gray = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)
_, color_thresh = cv2.threshold(color_diff_gray, 50, 255, cv2.THRESH_BINARY)

# ==========================================
# B. 흑백 차분 (Grayscale Difference)
# ==========================================
# 흑백 변환
gray_on = cv2.cvtColor(img_on, cv2.COLOR_BGR2GRAY)
gray_off = cv2.cvtColor(img_off, cv2.COLOR_BGR2GRAY)

# 차분 (Absolute Difference)
gray_diff = cv2.absdiff(gray_on, gray_off)

# 노이즈 제거 (Threshold)
_, gray_thresh = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)

# 결과 보여주기
print("\n결과 창이 떴습니다. 확인 후 아무 키나 누르면 종료됩니다.")
print("- 컬러 차분: 색상 정보를 유지한 채 차분 계산")
print("- 흑백 차분: 밝기만 비교하여 차분 계산")
show_resized("1. Laser ON", img_on)
show_resized("2-A. Color Diff (Raw)", color_diff)
show_resized("2-B. Grayscale Diff (Raw)", gray_diff)
show_resized("3-A. Color Result (Clean)", color_thresh)
show_resized("3-B. Grayscale Result (Clean)", gray_thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()