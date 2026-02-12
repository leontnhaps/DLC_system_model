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

# ==========================================
# 2. 마우스 콜백
# ==========================================
click_points = []
SCALE = 0.4  # 화면 표시 배율 (0.4배)

def mouse_callback(event, x, y, flags, param):
    """마우스 클릭 시 좌표 저장 및 표시"""
    if event == cv2.EVENT_LBUTTONDOWN:
        # 화면 좌표 → 원본 좌표 변환
        orig_x = int(x / SCALE)
        orig_y = int(y / SCALE)
        
        click_points.append((orig_x, orig_y))
        print(f"📍 Clicked: Screen({x}, {y}) -> Original({orig_x}, {orig_y})")
        
        # 클릭 지점 표시 (화면에 그리기)
        img_disp = param['img_disp']
        cv2.circle(img_disp, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(img_disp, f"({orig_x},{orig_y})", (x+5, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow(param['window_name'], img_disp)

# ==========================================
# 3. 메인 실행
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    current_dir = os.getcwd()
    
    print(">> 이미지 파일들을 선택해주세요 (다중 선택 가능)")
    file_paths = filedialog.askopenfilenames(
        title="레이저 위치 확인용 이미지 선택",
        initialdir=current_dir,
        filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")]
    )
    
    if not file_paths:
        print("❌ 파일 선택이 취소되었습니다.")
        exit()
        
    all_points = []
    
    print("\n[사용법]")
    print(f"- 이미지는 {SCALE}배 축소되어 표시됩니다.")
    print("- 마우스 왼쪽 클릭: 레이저 중심점 찍기")
    print("- ESC 키: 다음 이미지로 넘어가기 (저장됨)")
    print("- 'r' 키: 현재 이미지 초기화 (다시 찍기)")
    
    for i, path in enumerate(file_paths):
        filename = os.path.basename(path)
        print(f"\n[{i+1}/{len(file_paths)}] 열기: {filename}")
        
        img = imread_unicode(path)
        if img is None: continue
        
        # 화면 표시용 리사이즈
        h, w = img.shape[:2]
        new_w, new_h = int(w * SCALE), int(h * SCALE)
        img_disp = cv2.resize(img, (new_w, new_h))
        
        window_name = f"Check Laser Pos - {filename}"
        cv2.namedWindow(window_name)
        
        # 콜백 설정
        param = {'img_disp': img_disp.copy(), 'window_name': window_name}
        cv2.setMouseCallback(window_name, mouse_callback, param)
        
        click_points = []  # 현재 이미지 클릭 초기화
        
        while True:
            cv2.imshow(window_name, param['img_disp'])
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('r'):  # Reset
                # 다시 원본에서 리사이즈해서 초기화
                param['img_disp'] = cv2.resize(img, (new_w, new_h))
                click_points = []
                print("🔄 Reset")
        
        # ESC 누르면 현재까지 찍은 점들 저장
        if click_points:
            # 여러 번 찍었으면 마지막 점 사용 (또는 평균 사용 가능)
            last_pt = click_points[-1] 
            all_points.append(last_pt)
            print(f"✅ Saved: {last_pt}")
        
        cv2.destroyWindow(window_name)
    
    # ==========================================
    # 4. 결과 종합
    # ==========================================
    cv2.destroyAllWindows()
    
    if all_points:
        print("\n" + "="*40)
        print("🎯 최종 결과 (평균 좌표)")
        print("="*40)
        
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        
        avg_x = sum(xs) / len(xs)
        avg_y = sum(ys) / len(ys)
        
        print(f"총 {len(all_points)}개 이미지 측정됨")
        print(f"X 좌표들: {xs}")
        print(f"Y 좌표들: {ys}")
        print("-" * 20)
        print(f"🔥 평균 좌표: ({avg_x:.1f}, {avg_y:.1f})")
        print("="*40)
    else:
        print("\n❌ 측정된 좌표가 없습니다.")
