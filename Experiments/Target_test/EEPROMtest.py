import serial
import time

PORT = "COM5"      # 너 PC의 COM 번호로 수정
BAUD = 19200
OUT  = "battery_log.csv"

with serial.Serial(PORT, BAUD, timeout=2) as ser:
    time.sleep(1.0)            # 포트 열면 보드 리셋될 수 있어 잠깐 대기
    ser.write(b"D")            # 덤프 명령

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        while True:
            line = ser.readline().decode(errors="ignore")
            if not line:
                continue
            f.write(line)
            if line.strip() == "END":
                break

print("Saved:", OUT)