import serial
import time
import os
import re

PORT = "COM5"      # COM 번호 수정
BAUD = 9600

def get_next_filename():
    files = [f for f in os.listdir(".") if re.fullmatch(r"\d+\.csv", f)]
    nums = [int(f[:-4]) for f in files]
    next_num = max(nums, default=0) + 1
    return f"{next_num}.csv"

out_file = get_next_filename()

with serial.Serial(PORT, BAUD, timeout=2) as ser:
    time.sleep(1.5)   # 보드 리셋 대기
    ser.reset_input_buffer()
    
    # 1. 데이터 덤프 요청
    ser.write(b"D")
    print("Downloading data from Arduino...")

    # 2. CSV 파일 작성
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        while True:
            line = ser.readline().decode(errors="ignore")
            if not line:
                continue
            f.write(line)
            if line.strip() == "END":  # 데이터 전송 끝
                break

    print("Saved:", out_file)

    # 3. 저장이 완료된 후 초기화 명령어 'C' 전송
    print("Clearing Arduino EEPROM log...")
    ser.write(b"C")

    # 4. 아두이노에서 "OK,CLEARED" 응답 확인
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line == "OK,CLEARED":
            print("Log successfully cleared!")
            break
        elif not line: # 설정된 timeout(2초) 동안 응답이 없으면 탈출
            print("Warning: Clear command sent, but no confirmation received.")
            break