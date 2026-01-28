#!/usr/bin/env python3
"""
카메라 진단 스크립트
cv2.VideoCapture가 지원하는 해상도 테스트
"""

import cv2
import time

print("=" * 60)
print("카메라 진단 시작")
print("=" * 60)

# 테스트할 해상도들
resolutions = [
    (640, 480, "VGA"),
    (800, 600, "SVGA"),
    (1024, 768, "XGA"),
    (1280, 720, "HD 720p"),
    (1280, 1024, "SXGA"),
    (1920, 1080, "Full HD 1080p"),
    (2592, 1944, "5MP Max"),
]

for w, h, name in resolutions:
    print(f"\n테스트 중: {name} ({w}x{h})")
    
    camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    if not camera.isOpened():
        print(f"  ❌ 카메라 열기 실패")
        continue
    
    # 해상도 설정
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    
    # 실제 설정된 값 확인
    actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  설정: {w}x{h}")
    print(f"  실제: {actual_w}x{actual_h}")
    
    # 워밍업
    for _ in range(3):
        camera.read()
    time.sleep(0.1)
    
    # 실제 캡처 테스트
    success_count = 0
    for i in range(5):
        ret, frame = camera.read()
        if ret and frame is not None:
            success_count += 1
            if i == 0:
                print(f"  프레임 shape: {frame.shape}")
        else:
            print(f"  ❌ 캡처 {i+1} 실패")
    
    camera.release()
    
    if success_count == 5:
        print(f"  ✅ 성공! (5/5)")
    else:
        print(f"  ⚠️ 부분 성공 ({success_count}/5)")

print("\n" + "=" * 60)
print("진단 완료")
print("=" * 60)
