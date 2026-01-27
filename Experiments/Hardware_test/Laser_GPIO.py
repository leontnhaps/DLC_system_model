import RPi.GPIO as GPIO
import time

# 1. 핀 번호 설정 방식 선택 (BCM 모드 권장)
# BCM 모드는 핀의 기능 이름(GPIO 15)을 기준으로 합니다.
# 물리적 핀 10번 = BCM 15번
GPIO.setmode(GPIO.BCM)

# 2. 사용할 핀 번호 변수 설정
SIGNAL_PIN = 15 

# 3. 핀 설정 (출력 모드로 설정)
# 이 시점에서 핀의 전압이 제어 가능한 상태가 됩니다.
GPIO.setup(SIGNAL_PIN, GPIO.OUT)

print("테스트 시작: Ctrl+C를 누르면 종료됩니다.")

try:
    while True:
        # 신호 보내기 (HIGH = 3.3V) -> ESP32가 이를 감지하고 LED를 켬
        GPIO.output(SIGNAL_PIN, GPIO.HIGH)
        print("Signal: HIGH (LED ON)")
        time.sleep(1) # 1초 대기

        # 신호 끄기 (LOW = 0V) -> ESP32가 이를 감지하고 LED를 끔
        GPIO.output(SIGNAL_PIN, GPIO.LOW)
        print("Signal: LOW (LED OFF)")
        time.sleep(1) # 1초 대기

except KeyboardInterrupt:
    # Ctrl+C 눌렀을 때 안전하게 종료
    print("\n테스트 종료")
    GPIO.cleanup() # 핀 설정을 초기화하여 안전하게 만듦