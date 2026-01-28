#!/usr/bin/env python3
"""
Step 1: 카메라 기본 캡처 테스트
- Picamera2 초기화
- 이미지 캡처
- 파일 저장
"""
from picamera2 import Picamera2
import time
print("=" * 60)
print("카메라 초기화 중...")
print("=" * 60)
# 카메라 초기화
picam = Picamera2()
# Still 모드 설정 (고해상도)
config = picam.create_still_configuration(main={"size": (2592, 1944)})
picam.configure(config)
picam.start()
print("✅ 카메라 준비 완료")
print("   해상도: 2592x1944")
# 3A 수렴 대기 (Auto Exposure, Auto White Balance)
print("\n⏳ 노출/화이트밸런스 안정화 중...")
time.sleep(2.0)
# 캡처
print("📸 사진 촬영 중...")
picam.capture_file("test_capture.jpg")
print("✅ 저장 완료: test_capture.jpg")
print("\n" + "=" * 60)
print("테스트 성공!")
print("=" * 60)
picam.stop()