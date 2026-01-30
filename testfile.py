#!/usr/bin/env python3
from picamera2 import Picamera2
picam = Picamera2()
print("=" * 60)
print("카메라 센서 모드 정보")
print("=" * 60)
for i, mode in enumerate(picam.sensor_modes):
    print(f"\n모드 {i}:")
    print(f"  해상도: {mode['size']}")
    print(f"  비트: {mode.get('bit_depth', 'N/A')}")
    print(f"  포맷: {mode.get('format', 'N/A')}")
print("\n" + "=" * 60)
print("권장 설정")
print("=" * 60)
# 최대 해상도 찾기
max_mode = max(picam.sensor_modes, key=lambda m: m['size'][0] * m['size'][1])
max_w, max_h = max_mode['size']
max_mp = (max_w * max_h) / 1_000_000
print(f"최대 해상도: {max_w} x {max_h} ({max_mp:.1f}MP)")
picam.close()