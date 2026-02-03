#!/usr/bin/env python3
"""
Raspberrypi Agent - Complete with Manual Control
- ESP32 serial communication (Pan/Tilt)
- Manual control (move, LED, laser)
- Preview streaming
- Snap capture
- IR-CUT control
"""
import socket, struct, io, time, json, threading, glob
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import serial
# ==================== GPIO 설정 ====================
IR_CUT_PIN = 17  # BCM 17번
LASER_PIN = 15   # BCM 15번
# ==================== 서버 설정 ====================
SERVER_OPTIONS = {
    "1": ("192.168.0.9", "711a"),
    "2": ("172.30.1.13", "602a"),
    "3": ("10.190.176.118", "hotspot")
}
def select_server():
    print("\n" + "="*50)
    print("서버 선택")
    print("="*50)
    for key, (ip, name) in SERVER_OPTIONS.items():
        print(f"  [{key}] {name:10s} → {ip}")
    print("="*50)
    
    while True:
        choice = input("서버 번호 (1/2/3) [기본: 2]: ").strip()
        if not choice: choice = "2"
        if choice in SERVER_OPTIONS:
            ip, name = SERVER_OPTIONS[choice]
            print(f"✓ 선택: {name} ({ip})\n")
            return ip
SERVER_HOST = select_server()
CTRL_PORT = 7500
IMG_PORT = 7501
BAUD = 115200
# ==================== 시리얼 (ESP32) ====================
def open_serial():
    """ESP32 시리얼 연결"""
    cands = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyAMA0', '/dev/ttyS0'] + glob.glob('/dev/ttyUSB*')
    last_err = None
    for dev in cands:
        try:
            s = serial.Serial(dev, BAUD, timeout=0.2)
            print(f"[SERIAL] 연결: {dev}")
            return s
        except Exception as e:
            last_err = e
    raise RuntimeError(f"시리얼 연결 실패: {last_err}")
ser = open_serial()
def send_to_slave(obj: dict):
    """ESP32로 JSON 명령 전송"""
    ser.write((json.dumps(obj) + "\n").encode())
    ser.flush()
# ==================== 전역 변수 ====================
picam = None
preview_stop = threading.Event()
preview_thread = None
# ==================== 이미지 전송 ====================
def push_image(sock: socket.socket, name: str, data: bytes):
    header = struct.pack("<H", len(name.encode())) + name.encode() + struct.pack("<I", len(data))
    sock.sendall(header)
    sock.sendall(data)
# ==================== 프리뷰 워커 ====================
def preview_worker(img_sock, w=640, h=480, fps=5, q=70):
    global picam
    try:
        print(f"[PREVIEW] 시작: {w}x{h} @ {fps}fps, quality={q}")
        
        picam.stop()
        config = picam.create_video_configuration(main={"size": (w, h)})
        picam.configure(config)
        picam.options["quality"] = int(q)
        picam.start()
        time.sleep(0.1)
        
        interval = 1.0 / max(1, fps)
        frame_count = 0
        error_count = 0
        
        while not preview_stop.is_set():
            bio = None
            try:
                bio = io.BytesIO()
                picam.capture_file(bio, format="jpeg")
                jpeg_data = bio.getvalue()
                
                push_image(img_sock, f"_preview_{int(time.time()*1000)}.jpg", jpeg_data)
                frame_count += 1
                error_count = 0  # 성공 시 에러 카운트 리셋
                
                # 주기적 로그 (100프레임마다)
                if frame_count % 100 == 0:
                    print(f"[PREVIEW] {frame_count} 프레임 전송 완료")
                
            except Exception as e:
                error_count += 1
                print(f"[PREVIEW] 프레임 오류 ({error_count}): {e}")
                
                # 연속 10회 오류 시 중단
                if error_count >= 10:
                    print("[PREVIEW] 오류 과다 → 중단")
                    break
            
            finally:
                # 메모리 누수 방지 - BytesIO 명시적 닫기
                if bio:
                    bio.close()
            
            time.sleep(interval)
        
        print(f"[PREVIEW] 중지됨 (총 {frame_count} 프레임)")
        
    except Exception as e:
        print(f"[PREVIEW] 치명적 오류: {e}")
    
    finally:
        # 카메라 중지
        try:
            picam.stop()
        except:
            pass
# ==================== 스냅 캡처 ====================
def snap_capture(img_sock, w, h, q, name):
    """고해상도 스틸 이미지 캡처"""
    global picam
    bio = None
    try:
        print(f"[SNAP] 캡처 중: {w}x{h}, quality={q}")
        
        # 프리뷰 중지
        preview_stop.set()
        if preview_thread and preview_thread.is_alive():
            preview_thread.join(timeout=0.7)
        
        # Still 모드로 전환
        picam.stop()
        config = picam.create_still_configuration(main={"size": (w, h)})
        picam.configure(config)
        picam.options["quality"] = int(q)
        picam.start()
        time.sleep(0.2)
        
        # 캡처
        bio = io.BytesIO()
        picam.capture_file(bio, format="jpeg")
        jpeg_data = bio.getvalue()
        
        # 전송
        push_image(img_sock, name, jpeg_data)
        print(f"[SNAP] 완료: {name} ({len(jpeg_data)} bytes)")
        
    except Exception as e:
        print(f"[SNAP] 오류: {e}")
    
    finally:
        # 메모리 누수 방지
        if bio:
            bio.close()
        
        # 카메라 중지
        try:
            picam.stop()
        except:
            pass
# ==================== 메인 ====================
def main():
    global picam, preview_thread
    
    # GPIO 초기화
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(IR_CUT_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(LASER_PIN, GPIO.OUT, initial=GPIO.LOW)
    print("[GPIO] 초기화 완료 (IR-CUT: GPIO 17, Laser: GPIO 15)")
    
    # 카메라 초기화
    print("=" * 60)
    print("카메라 초기화")
    print("=" * 60)
    picam = Picamera2()
    print("✅ 카메라 준비 완료")
    
    # 서버 연결
    print(f"\n서버 연결: {SERVER_HOST}:{IMG_PORT}")
    while True:
        try:
            img_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            img_sock.connect((SERVER_HOST, IMG_PORT))
            print("✅ IMG 소켓 연결!")
            break
        except Exception as e:
            print(f"⏳ 재시도... ({e})")
            time.sleep(1.0)
    
    print(f"제어 소켓 연결: {SERVER_HOST}:{CTRL_PORT}")
    while True:
        try:
            ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ctrl_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            ctrl_sock.connect((SERVER_HOST, CTRL_PORT))
            print("✅ CTRL 소켓 연결!")
            break
        except Exception as e:
            print(f"⏳ 재시도... ({e})")
            time.sleep(1.0)
    
    print("\n" + "=" * 60)
    print("명령 대기 중...")
    print("=" * 60)
    
    # 명령 수신 루프
    buf = b""
    try:
        while True:
            data = ctrl_sock.recv(4096)
            if not data: break
            buf += data
            
            while True:
                nl = buf.find(b"\n")
                if nl < 0: break
                line = buf[:nl].decode("utf-8", "ignore").strip()
                buf = buf[nl+1:]
                if not line: continue
                
                try:
                    cmd = json.loads(line)
                except:
                    continue
                
                c = cmd.get("cmd")
                
                if c == "preview":
                    enable = cmd.get("enable", True)
                    if enable:
                        preview_stop.set()
                        if preview_thread and preview_thread.is_alive():
                            preview_thread.join(timeout=0.5)
                        
                        preview_stop.clear()
                        w = int(cmd.get("width", 640))
                        h = int(cmd.get("height", 480))
                        fps = int(cmd.get("fps", 5))
                        q = int(cmd.get("quality", 70))
                        
                        preview_thread = threading.Thread(
                            target=preview_worker,
                            args=(img_sock, w, h, fps, q),
                            daemon=True
                        )
                        preview_thread.start()
                        print(f"[CMD] 프리뷰 시작: {w}x{h}")
                    else:
                        preview_stop.set()
                        if preview_thread:
                            preview_thread.join(timeout=0.7)
                        print("[CMD] 프리뷰 중지")
                
                elif c == "snap":
                    w = int(cmd.get("width", 2592))
                    h = int(cmd.get("height", 1944))
                    q = int(cmd.get("quality", 95))
                    name = cmd.get("save", "snap.jpg")
                    
                    threading.Thread(
                        target=snap_capture,
                        args=(img_sock, w, h, q, name),
                        daemon=True
                    ).start()
                    print(f"[CMD] Snap 요청: {w}x{h}")
                
                elif c == "move":
                    # Pan/Tilt 이동
                    pan = float(cmd.get("pan", 0.0))
                    tilt = float(cmd.get("tilt", 0.0))
                    speed = int(cmd.get("speed", 100))
                    acc = float(cmd.get("acc", 1.0))
                    
                    send_to_slave({
                        "T": 133,
                        "X": pan,
                        "Y": tilt,
                        "SPD": speed,
                        "ACC": acc
                    })
                    print(f"[MOVE] Pan={pan}, Tilt={tilt}, Speed={speed}, Acc={acc}")
                
                elif c == "led":
                    # LED 제어
                    val = int(cmd.get("value", 0))
                    send_to_slave({
                        "T": 132,
                        "IO4": val,
                        "IO5": val
                    })
                    print(f"[LED] Value={val}")
                
                elif c == "laser":
                    # 레이저 제어
                    val = int(cmd.get("value", 0))
                    GPIO.output(LASER_PIN, GPIO.HIGH if val else GPIO.LOW)
                    print(f"[LASER] {'ON' if val else 'OFF'}")
                
                elif c == "ir_cut":
                    # IR-CUT 제어
                    mode = cmd.get("mode", "day")
                    if mode == "day":
                        GPIO.output(IR_CUT_PIN, GPIO.HIGH)
                        print("[IR-CUT] Day Mode (IR 필터 ON)")
                    else:
                        GPIO.output(IR_CUT_PIN, GPIO.LOW)
                        print("[IR-CUT] Night Mode (IR 필터 OFF)")
                
                else:
                    print(f"[CMD] 알 수 없는 명령: {c}")
    
    finally:
        GPIO.cleanup()
        print("[GPIO] Cleanup 완료")
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[MAIN] 종료")
    finally:
        GPIO.cleanup()