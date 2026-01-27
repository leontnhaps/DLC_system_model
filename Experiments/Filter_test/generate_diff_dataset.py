#!/usr/bin/env python3
"""
차분 이미지 데이터셋 생성기 (YOLO 학습용)
- 선택한 폴더 내의 '_ud' (보정된) 이미지들을 스캔합니다.
- 같은 Pan/Tilt 위치의 LED ON/OFF 쌍을 찾습니다.
- 두 이미지의 차분(Difference) 이미지를 생성하여 'diff_dataset' 폴더에 저장합니다.
"""

import cv2
import numpy as np
import os
import re
from tkinter import Tk, filedialog
from pathlib import Path

# 파일명 파싱용 정규표현식
# 예: img_t+00_p+000_..._led_on.ud.jpg
# t값, p값, on/off 상태를 추출합니다.
PATTERN = re.compile(r"img_t([+\-]\d+)_p([+\-]\d+)_.*_led_(on|off)\.ud\.(jpg|png|jpeg)", re.IGNORECASE)

def load_image(path):
    """한글 경로 지원 이미지 로드"""
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(stream, cv2.IMREAD_COLOR)
    except Exception:
        return None

def save_image(path, img):
    """한글 경로 지원 이미지 저장"""
    try:
        ext = os.path.splitext(path)[1]
        result, encoded_img = cv2.imencode(ext, img)
        if result:
            with open(path, "wb") as f:
                encoded_img.tofile(f)
            return True
    except Exception as e:
        print(f"저장 실패: {e}")
    return False

def main():
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("   차분 이미지 데이터셋 생성기   ")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("1. '.ud' (보정된) 이미지가 들어있는 폴더를 선택하세요.")
    print("   (예: captures_gui_2024...)")
    
    # 폴더 선택
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    in_dir = filedialog.askdirectory(title="입력 폴더 선택 (captures_gui_...)")
    root.destroy()

    if not in_dir:
        print("❌ 폴더 선택이 취소되었습니다.")
        return

    in_path = Path(in_dir)
    out_path = in_path / "diff_dataset"
    out_path.mkdir(exist_ok=True)
    
    print(f"\n📂 입력: {in_path}")
    print(f"📂 출력: {out_path}")

    # 파일 그룹핑
    # key: (tilt, pan), value: {'on': file, 'off': file}
    pairs = {}
    
    # .ud가 붙은 이미지 파일 검색
    files = list(in_path.glob("*.ud.*"))
    print(f"🔍 발견된 보정 이미지(.ud) 수: {len(files)}")

    if len(files) == 0:
        print("⚠️ '.ud'가 포함된 이미지 파일이 없습니다. 보정 옵션을 켜고 스캔했는지 확인하세요.")
        return

    for f in files:
        match = PATTERN.search(f.name)
        if match:
            t_val = int(match.group(1))
            p_val = int(match.group(2))
            state = match.group(3).lower() # on or off
            
            key = (t_val, p_val)
            if key not in pairs:
                pairs[key] = {}
            
            pairs[key][state] = f

    # 차분 이미지 생성
    print("\n🚀 차분 이미지 생성 시작...")
    count = 0
    skip = 0
    
    # 진행률 표시를 위해 전체 키 정렬
    sorted_keys = sorted(pairs.keys())
    
    for (t, p) in sorted_keys:
        group = pairs[(t, p)]
        
        if 'on' in group and 'off' in group:
            f_on = group['on']
            f_off = group['off']
            
            img_on = load_image(f_on)
            img_off = load_image(f_off)
            
            if img_on is None or img_off is None:
                print(f"⚠️ 읽기 실패: (T{t}, P{p})")
                skip += 1
                continue
                
            # 크기 검사
            if img_on.shape != img_off.shape:
                print(f"⚠️ 크기 불일치: (T{t}, P{p}) - 리사이징 수행")
                img_off = cv2.resize(img_off, (img_on.shape[1], img_on.shape[0]))
            
            # 절대 차분 계산
            diff = cv2.absdiff(img_on, img_off)
            
            # 저장 파일명: diff_t+00_p+000.jpg
            out_name = f"diff_t{t:+03d}_p{p:+04d}.jpg"
            save_path = out_path / out_name
            
            if save_image(save_path, diff):
                count += 1
                print(f"[{count}] 생성: {out_name}")
            else:
                skip += 1
        else:
            # 짝이 안 맞는 경우
            skip += 1
            # print(f"⚠️ 짝 없음: (T{t}, P{p}) -> {list(group.keys())}")

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"✅ 완료!")
    print(f"   - 생성된 이미지: {count}장")
    print(f"   - 건너뜀 (짝 없음/오류): {skip}장")
    print(f"   - 저장 폴더: {out_path}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

if __name__ == "__main__":
    main()
