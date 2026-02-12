#!/usr/bin/env python3
"""
Raspberrypi Agent - Complete with Manual Control + Scan + Exposure Control
- ESP32 serial communication (Pan/Tilt)
- Manual control (move, LED, laser)
- Preview streaming
- Snap capture (Exposure control supported)
- IR-CUT control
- Grid Scan (Boustrophedon pattern)
Modified: 2026-02-12
- Added exposure control (shutter_speed, analogue_gain) to snap command
"""
import socket, struct, io, time, json, threading, glob
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import serial

# ==================== GPIO 설정 ====================
IR_CUT_PIN = 17  # BCM 17번
LASER_PIN = 23   # BCM 23번

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
scan_stop = threading.Event()
scan_thread = None

# ==================== 이미지 전송 ====================
def push_image(sock: socket.socket, name: str, data: bytes):
    header = struct.pack("<H", len(name.encode())) + name.encode() + struct.pack("<I", len(data))
    sock.sendall(header)
    sock.sendall(data)

# ==================== Scan 유틸리티 ====================
def generate_boustrophedon_grid(pan_min, pan_max, pan_step, 
                                 tilt_min, tilt_max, tilt_step):
    """Boustrophedon 패턴 (뱀 모양) 그리드 생성"""
    grid = []
    tilt_values = list(range(tilt_min, tilt_max + 1, tilt_step))
    for tilt_idx, tilt in enumerate(tilt_values):
        pan_values = list(range(pan_min, pan_max + 1, pan_step))
        if tilt_idx % 2 == 1:
            pan_values.reverse()
        for pan in pan_values:
            grid.append((pan, tilt))
    return grid

# ==================== Scan Worker ====================
def scan_worker(cmd, ctrl_sock, img_sock):
    """Grid scan worker"""
    global picam
    
    try:
        # 파라미터 추출
        pan_min = int(cmd["pan_min"])
        pan_max = int(cmd["pan_max"])
        pan_step = int(cmd["pan_step"])
        tilt_min = int(cmd["tilt_min"])
        tilt_max = int(cmd["tilt_max"])
        tilt_step = int(cmd["tilt_step"])
        
        speed = int(cmd["speed"])
        acc = float(cmd["acc"])
        settle = float(cmd["settle"])
        led_settle = float(cmd.get("led_settle", 0.4))
        
        width = int(cmd["width"])
        height = int(cmd["height"])
        quality = int(cmd["quality"])
        session = cmd["session"]
        
        print(f"[SCAN] 시작: Pan[{pan_min}~{pan_max}:{pan_step}], Tilt[{tilt_min}~{tilt_max}:{tilt_step}]")
        
        # Grid 생성
        grid = generate_boustrophedon_grid(
            pan_min, pan_max, pan_step,
            tilt_min, tilt_max, tilt_step
        )
        total = len(grid)
        
        # Start 이벤트 전송
        ctrl_sock.sendall((json.dumps({
            "event": "start",
            "session": session,
            "total": total
        }) + "\n").encode())
        
        # 카메라 설정 (Scan은 Auto Exposure 사용 - YOLO 인식률 위해)
        picam.stop()
        config = picam.create_still_configuration(main={"size": (width, height)})
        picam.configure(config)
        picam.options["quality"] = quality
        picam.start()
        time.sleep(0.2)
        
        for i, (pan, tilt) in enumerate(grid):
            if scan_stop.is_set():
                break
            
            print(f"[SCAN] [{i+1}/{total}] Pan={pan}, Tilt={tilt}")
            
            # ① 이동
            send_to_slave({"T": 133, "X": float(pan), "Y": float(tilt), "SPD": speed, "ACC": acc})
            time.sleep(settle)
            
            # ② LED ON
            send_to_slave({"T": 132, "IO4": 255, "IO5": 255})
            time.sleep(led_settle)
            
            # ③ 촬영 (ON)
            bio = io.BytesIO()
            picam.capture_file(bio, format="jpeg")
            jpeg_data = bio.getvalue()
            bio.close()
            push_image(img_sock, f"{session}_t{tilt:+03d}_p{pan:+04d}_led_on.jpg", jpeg_data)
            
            # ④ LED OFF
            send_to_slave({"T": 132, "IO4": 0, "IO5": 0})
            time.sleep(led_settle)
            
            # ⑤ 촬영 (OFF)
            bio = io.BytesIO()
            picam.capture_file(bio, format="jpeg")
            jpeg_data = bio.getvalue()
            bio.close()
            name_off = f"{session}_t{tilt:+03d}_p{pan:+04d}_led_off.jpg"
            push_image(img_sock, name_off, jpeg_data)
            
            # ⑥ Progress
            ctrl_sock.sendall((json.dumps({
                "event": "progress",
                "done": i + 1,
                "total": total,
                "name": name_off
            }) + "\n").encode())
        
        ctrl_sock.sendall((json.dumps({"event": "done"}) + "\n").encode())
        print(f"[SCAN] 완료")
        
    except Exception as e:
        print(f"[SCAN] 오류: {e}")
        ctrl_sock.sendall((json.dumps({"event": "error", "message": str(e)}) + "\n").encode())
    
    finally:
        try:
            send_to_slave({"T": 132, "IO4": 0, "IO5": 0})
        except:
            pass

# ==================== 프리뷰 워커 ====================
def preview_worker(img_sock, w=640, h=480, fps=5, q=70):
    global picam
    try:
        print(f"[PREVIEW] 시작: {w}x{h} @ {fps}fps")
        picam.stop()
        config = picam.create_video_configuration(main={"size": (w, h)})
        picam.configure(config)
        picam.options["quality"] = int(q)
        picam.start()
        time.sleep(0.1)
        
        interval = 1.0 / max(1, fps)
        
        while not preview_stop.is_set():
            bio = io.BytesIO()
            picam.capture_file(bio, format="jpeg")
            push_image(img_sock, f"_preview_{int(time.time()*1000)}.jpg", bio.getvalue())
            bio.close()
            time.sleep(interval)
            
    except Exception as e:
        print(f"[PREVIEW] 오류: {e}")
    finally:
        try:
            picam.stop()
        except:
            pass

# ==================== 스냅 캡처 (노출 제어 추가) ====================
def snap_capture(img_sock, w, h, q, name, shutter_speed=None, analogue_gain=None):
    """
    고해상도 스틸 이미지 캡처 (노출 제어 가능)
    shutter_speed: 노출 시간 (microseconds), None이면 Auto
    analogue_gain: 아날로그 게인 (1.0 ~ 16.0 등), None이면 Auto
    """
    global picam
    bio = None
    try:
        print(f"[SNAP] 캡처 중: {w}x{h}, quality={q}")
        if shutter_speed:
            print(f"       👉 노출 제어: Shutter={shutter_speed}µs, Gain={analogue_gain}")
        
        # 프리뷰 중지
        preview_stop.set()
        if preview_thread and preview_thread.is_alive():
            preview_thread.join(timeout=0.7)
        
        # Still 모드로 전환 + 노출 제어
        picam.stop()
        
        # Controls 설정
        controls = {}
        if shutter_speed is not None:
             # 노출 시간 (µs)
            controls["ExposureTime"] = int(shutter_speed)
        if analogue_gain is not None:
             # 아날로그 게인
            controls["AnalogueGain"] = float(analogue_gain)
            
        # 프레임 지속 시간 설정 (노출 시간보다 길어야 함)
        if shutter_speed:
            min_duration = int(shutter_speed) + 5000  # 여유 5ms
            controls["FrameDurationLimits"] = (min_duration, 2000000) # 최대 2초
            
        config = picam.create_still_configuration(main={"size": (w, h)}, controls=controls)
        picam.configure(config)
        picam.options["quality"] = int(q)
        picam.start()
        
        # 노출 안정화 대기 (수동 노출은 더 짧아도 됨)
        # 하지만 Auto일 경우를 대비해 넉넉히
        wait_time = 0.5 if shutter_speed is None else 0.2
        time.sleep(wait_time)
        
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
        if bio: bio.close()
        try:
            picam.stop()
        except:
            pass

# ==================== 메인 ====================
def main():
    global picam, preview_thread, scan_thread
    
    # GPIO 초기화
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(IR_CUT_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(LASER_PIN, GPIO.OUT, initial=GPIO.LOW)
    print("[GPIO] 초기화 완료")
    
    # 카메라 초기화
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
            time.sleep(1.0)
    
    print("\n" + "=" * 60)
    print("명령 대기 중...")
    print("=" * 60)
    
    # 명령 수신
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
                        print(f"[CMD] 프리뷰 시작")
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
                    
                    # ⭐ 노출 제어 파라미터 추가
                    shutter = cmd.get("shutter_speed") # None or int (µs)
                    gain = cmd.get("analogue_gain")    # None or float
                    
                    threading.Thread(
                        target=snap_capture,
                        args=(img_sock, w, h, q, name, shutter, gain),
                        daemon=True
                    ).start()
                    print(f"[CMD] Snap 요청 {'(Exposure Controlled)' if shutter else ''}")
                
                elif c == "scan_run":
                    scan_stop.set()
                    if scan_thread and scan_thread.is_alive():
                        scan_thread.join(timeout=1.0)
                    scan_stop.clear()
                    scan_thread = threading.Thread(
                        target=scan_worker,
                        args=(cmd, ctrl_sock, img_sock),
                        daemon=True
                    )
                    scan_thread.start()
                    print("[CMD] Scan 시작")
                
                elif c == "scan_stop":
                    scan_stop.set()
                    if scan_thread:
                        scan_thread.join(timeout=1.0)
                    print("[CMD] Scan 중지")
                
                elif c == "move":
                    pan = float(cmd.get("pan", 0.0))
                    tilt = float(cmd.get("tilt", 0.0))
                    speed = int(cmd.get("speed", 100))
                    acc = float(cmd.get("acc", 1.0))
                    send_to_slave({"T": 133, "X": pan, "Y": tilt, "SPD": speed, "ACC": acc})
                
                elif c == "led":
                    val = int(cmd.get("value", 0))
                    send_to_slave({"T": 132, "IO4": val, "IO5": val})
                
                elif c == "laser":
                    val = int(cmd.get("value", 0))
                    GPIO.output(LASER_PIN, GPIO.HIGH if val else GPIO.LOW)
                
                elif c == "ir_cut":
                    mode = cmd.get("mode", "day")
                    GPIO.output(IR_CUT_PIN, GPIO.LOW if mode == "day" else GPIO.HIGH)
    
    except KeyboardInterrupt:
        print("\n[EXIT] 종료 중...")
    finally:
        preview_stop.set()
        scan_stop.set()
        GPIO.cleanup()
        try: ctrl_sock.close()
        except: pass
        try: img_sock.close()
        except: pass

if __name__ == "__main__":
    main()
