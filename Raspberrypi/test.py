#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pi Agent (NEW IR-CUT Camera) — cv2.VideoCapture 기반 카메라 제어
- 새 RPi IR-CUT Camera (B) 전용
- cv2.VideoCapture (V4L2) 사용으로 빠른 캡처
- 프리뷰/스캔/스냅 통합
"""

import os, io, json, time, glob, threading, datetime, socket, struct
import serial
import cv2
import numpy as np
import RPi.GPIO as GPIO

# ===================== GPIO 설정 =====================
LASER_PIN = 15  # BCM 15번 핀 (물리적 핀 10번)
IR_CUT_PIN = 17  # BCM 17번 핀 (IR-CUT 필터 제어, 예시)

# ===================== 환경 설정 =====================
SERVER_OPTIONS = {
    "1": ("192.168.0.9", "711a"),
    "2": ("172.30.1.13", "602a"),
    "3": ("10.190.176.118", "hotspot")
}

def select_server():
    """실행 시 서버 IP 선택"""
    print("\n" + "="*50)
    print("서버 선택 (Server Selection)")
    print("="*50)
    for key, (ip, name) in SERVER_OPTIONS.items():
        print(f"  [{key}] {name:10s} → {ip}")
    print("="*50)
    
    while True:
        choice = input("서버 번호를 선택하세요 (1/2/3) [기본값: 2]: ").strip()
        if not choice:
            choice = "2"
        if choice in SERVER_OPTIONS:
            ip, name = SERVER_OPTIONS[choice]
            print(f"✓ 선택됨: {name} ({ip})\n")
            return ip
        else:
            print("❌ 잘못된 입력입니다. 1, 2, 3 중에서 선택하세요.")

SERVER_HOST = os.getenv("SERVER_HOST") or select_server()
CTRL_PORT = int(os.getenv("CTRL_PORT", "7500"))
IMG_PORT = int(os.getenv("IMG_PORT", "7501"))
BAUD = 115200

MAX_W, MAX_H = 2592, 1944  # 센서 최대

# ===================== 시리얼(ESP32) =====================
def open_serial():
    cands = ['/dev/ttyUSB0','/dev/ttyUSB1','/dev/ttyAMA0','/dev/ttyS0'] + glob.glob('/dev/ttyUSB*')
    last=None
    for dev in cands:
        try:
            s = serial.Serial(dev, BAUD, timeout=0.2)
            print(f"[SERIAL] Connected: {dev}")
            return s
        except Exception as e:
            last=e
    raise RuntimeError(f"Serial open failed: {last}")

ser = open_serial()

def send_to_slave(obj: dict):
    ser.write((json.dumps(obj) + "\n").encode()); ser.flush()

# ===================== 카메라 (cv2.VideoCapture) =====================
camera = None
cam_lock = threading.Lock()
current_width = 640
current_height = 480

def init_camera(w=640, h=480):
    """카메라 초기화 또는 해상도 변경"""
    global camera, current_width, current_height
    
    with cam_lock:
        if camera is not None:
            camera.release()
        
        camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        current_width = w
        current_height = h
        
        # 워밍업 (첫 프레임 버리기)
        for _ in range(3):
            camera.read()
        
        print(f"[CAMERA] Initialized: {w}x{h}")

def capture_frame(quality=90):
    """현재 설정으로 프레임 캡처 → JPEG bytes 반환"""
    with cam_lock:
        ret, frame = camera.read()
        if not ret or frame is None:
            raise RuntimeError("Camera read failed")
        
        # JPEG 인코딩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ret, jpeg = cv2.imencode('.jpg', frame, encode_param)
        if not ret:
            raise RuntimeError("JPEG encoding failed")
        
        return jpeg.tobytes()

# ===================== 공통 유틸 =====================
def irange(a,b,s):
    if s<=0: raise ValueError("step>0")
    out=[]; v=a
    if a<=b:
        while v<=b: out.append(v); v+=s
    else:
        while v>=b: out.append(v); v-=s
    return out

def push_image(sock: socket.socket, name: str, data: bytes):
    """GUI 이미지 채널 프로토콜로 이미지 전송"""
    header = struct.pack("<H", len(name.encode())) + name.encode() + struct.pack("<I", len(data))
    sock.sendall(header); sock.sendall(data)

# ===================== 프리뷰 스레드 =====================
preview_thread = None
preview_stop = threading.Event()
preview_running = threading.Event()

def preview_worker(img_sock: socket.socket, w=640, h=480, fps=5, q=70):
    try:
        # 프리뷰는 최대 1920x1080로 제한 (V4L2 한계)
        w = min(w, 1920)
        h = min(h, 1080)
        
        init_camera(w, h)
        preview_running.set()
        interval = 1.0/max(1,fps)
        
        while not preview_stop.is_set():
            try:
                jpeg_bytes = capture_frame(quality=q)
                push_image(img_sock, f"_preview_{int(time.time()*1000)}.jpg", jpeg_bytes)
            except Exception as e:
                print(f"[PREVIEW] capture err: {e}")
                break  # 에러 시 루프 종료
            
            time.sleep(interval)
    except Exception as e:
        print(f"[PREVIEW] err: {e}")
    finally:
        preview_running.clear()

def preview_start(img_sock, w=640, h=480, fps=5, q=70):
    global preview_thread
    preview_stop.set()
    if preview_thread and preview_thread.is_alive():
        preview_thread.join(timeout=0.5)
    preview_stop.clear()
    
    # 프리뷰 해상도 제한
    w = min(w, 1920)
    h = min(h, 1080)
    
    preview_thread = threading.Thread(target=preview_worker, args=(img_sock,w,h,fps,q), daemon=True)
    preview_thread.start()

def preview_stop_now():
    preview_stop.set()
    if preview_thread and preview_thread.is_alive():
        preview_thread.join(timeout=0.7)

# ===================== 스캔 =====================
scan_stop_evt = threading.Event()

def scan_worker(params, ctrl_sock: socket.socket, img_sock: socket.socket):
    try:
        # 프리뷰 멈춤
        preview_stop_now()

        w = int(params.get("width", MAX_W))
        h = int(params.get("height", MAX_H))
        q = int(params.get("quality", 90))
        
        # 고해상도 캡처 방식 결정
        use_libcamera = (w > 1920 or h > 1080)
        
        if not use_libcamera:
            # cv2로 가능한 해상도
            init_camera(w, h)
            time.sleep(0.2)
        else:
            print(f"[SCAN] Using libcamera-still for {w}x{h}")

        pans  = irange(int(params["pan_min"]),  int(params["pan_max"]),  int(params["pan_step"]))
        tilts = irange(int(params["tilt_min"]), int(params["tilt_max"]), int(params["tilt_step"]))
        num_positions = len(pans)*len(tilts)
        total = num_positions * 2
        speed  = int(params.get("speed",100))
        acc    = float(params.get("acc",1.0))
        settle = float(params.get("settle",0.25))
        led_settle = float(params.get("led_settle",0.15))

        def send_evt(obj):
            try: ctrl_sock.sendall((json.dumps(obj,separators=(",",":"))+"\n").encode())
            except: pass
        
        def capture_scan_image():
            """스캔용 이미지 캡처 (해상도에 따라 방식 선택)"""
            if use_libcamera:
                # libcamera-still 사용
                import subprocess
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    subprocess.run([
                        "libcamera-still",
                        "-o", tmp_path,
                        "--width", str(w),
                        "--height", str(h),
                        "-q", str(q),
                        "-t", "1",
                        "-n"  # no preview
                    ], check=True, capture_output=True)
                    
                    with open(tmp_path, 'rb') as f:
                        jpeg_bytes = f.read()
                    os.unlink(tmp_path)
                    return jpeg_bytes
                except Exception as e:
                    print(f"[SCAN] libcamera-still error: {e}")
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    raise
            else:
                # cv2 사용
                return capture_frame(quality=q)

        send_evt({"event":"start","total":total})
        done=0

        # 첫 위치로 이동
        first_pan = pans[0]
        first_tilt = tilts[0]
        send_to_slave({"T":133, "X": float(first_pan), "Y": float(first_tilt), "SPD": 100, "ACC": 1.0})
        time.sleep(2.0)

        for i,t in enumerate(tilts):
            row = pans if i%2==0 else list(reversed(pans))
            for p in row:
                if scan_stop_evt.is_set():
                    raise InterruptedError
                
                # 이동
                send_to_slave({"T":133, "X": float(p), "Y": float(t), "SPD": speed, "ACC": acc})
                time.sleep(settle)

                # === LED ON → 촬영 ===
                send_to_slave({"T":132, "IO4": 255, "IO5": 255})
                time.sleep(led_settle)
                
                jpeg_on = capture_frame(quality=q)
                
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                name_on = f"img_t{t:+03d}_p{p:+04d}_{ts}_led_on.jpg"
                push_image(img_sock, name_on, jpeg_on)
                
                done += 1
                send_evt({"event":"progress","done":done,"total":total,"name":name_on})

                # === LED OFF → 촬영 ===
                send_to_slave({"T":132, "IO4": 0, "IO5": 0})
                time.sleep(led_settle)
                
                jpeg_off = capture_frame(quality=q)
                
                ts2 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                name_off = f"img_t{t:+03d}_p{p:+04d}_{ts2}_led_off.jpg"
                push_image(img_sock, name_off, jpeg_off)
                
                done += 1
                send_evt({"event":"progress","done":done,"total":total,"name":name_off})

        send_to_slave({"T":135})
        send_evt({"event":"done"})
    except InterruptedError:
        send_to_slave({"T":135})
        try: ctrl_sock.sendall((json.dumps({"event":"aborted"})+"\n").encode())
        except: pass
    except Exception as e:
        try: ctrl_sock.sendall((json.dumps({"event":"error","message":str(e)})+"\n").encode())
        except: pass
    finally:
        scan_stop_evt.clear()

# ===================== 스냅(한 장 캡처) =====================
def snap_once(cmd: dict, img_sock: socket.socket, ctrl_sock: socket.socket):
    try:
        # 프리뷰 정지
        preview_stop.set()
        global preview_thread
        if preview_thread and preview_thread.is_alive():
            preview_thread.join(timeout=0.8)

        W = int(cmd.get("width",  MAX_W))
        H = int(cmd.get("height", MAX_H))
        Q = int(cmd.get("quality", 90))
        fname = cmd.get("save") or datetime.datetime.now().strftime("snap_%Y%m%d_%H%M%S.jpg")

        # 스냅용 해상도 설정
        init_camera(W, H)
        time.sleep(0.1)

        # 실제 스냅
        jpeg_bytes = capture_frame(quality=Q)

        # IMG 채널로 전송
        push_image(img_sock, fname, jpeg_bytes)

        # 완료 이벤트
        try:
            ctrl_sock.sendall((json.dumps({"event":"snap_done","name":fname,"size":len(jpeg_bytes)})+"\n").encode())
        except:
            pass

    except Exception as e:
        try:
            ctrl_sock.sendall((json.dumps({"event":"error","where":"snap","message":str(e)})+"\n").encode())
        except:
            pass

# ===================== 메인: PC 서버에 접속 =====================
def main():
    # 카메라 초기화
    init_camera(640, 480)
    
    # 이미지 소켓
    while True:
        try:
            img = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            img.connect((SERVER_HOST, IMG_PORT))
            print(f"[IMG] connected {SERVER_HOST}:{IMG_PORT}")
            break
        except Exception as e:
            print(f"[IMG] retry: {e}"); time.sleep(1.0)

    # 제어 소켓
    while True:
        try:
            ctrl = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ctrl.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            ctrl.connect((SERVER_HOST, CTRL_PORT))
            print(f"[CTRL] connected {SERVER_HOST}:{CTRL_PORT}")
            break
        except Exception as e:
            print(f"[CTRL] retry: {e}"); time.sleep(1.0)

    # 제어 수신 루프
    buf=b""
    while True:
        data = ctrl.recv(4096)
        if not data: break
        buf += data
        while True:
            nl = buf.find(b"\n")
            if nl < 0: break
            line = buf[:nl].decode("utf-8","ignore").strip()
            buf = buf[nl+1:]
            if not line: continue
            try:
                cmd = json.loads(line)
            except:
                continue

            c = cmd.get("cmd")
            if c == "scan_run":
                threading.Thread(target=scan_worker, args=(cmd, ctrl, img), daemon=True).start()
            elif c == "scan_stop":
                scan_stop_evt.set()
            elif c == "move":
                send_to_slave({"T":133, "X": float(cmd.get("pan",0.0)), "Y": float(cmd.get("tilt",0.0)),
                               "SPD": int(cmd.get("speed",100)), "ACC": float(cmd.get("acc",1.0))})
            elif c == "led":
                val = int(cmd.get("value",0))
                send_to_slave({"T":132, "IO4": val, "IO5": val})
            elif c == "preview":
                enable = bool(cmd.get("enable", True))
                if enable:
                    w=int(cmd.get("width",640)); h=int(cmd.get("height",480))
                    fps=int(cmd.get("fps",5)); q=int(cmd.get("quality",70))
                    preview_start(img, w,h,fps,q)
                else:
                    preview_stop_now()
            elif c == "snap":
                snap_once(cmd, img, ctrl)
            elif c == "laser":
                val = int(cmd.get("value", 0))
                GPIO.output(LASER_PIN, GPIO.HIGH if val else GPIO.LOW)
                print(f"[LASER] {'ON' if val else 'OFF'}")
            elif c == "ir_cut":
                # IR-CUT 필터 제어 (새 기능)
                val = int(cmd.get("value", 0))
                GPIO.output(IR_CUT_PIN, GPIO.HIGH if val else GPIO.LOW)
                print(f"[IR-CUT] {'Day Mode' if val else 'Night Mode'}")
            else:
                pass

    # 종료 시 정리
    if camera:
        camera.release()
    GPIO.cleanup()
    print("[CLEANUP] Done")

if __name__ == "__main__":
    # GPIO 초기화
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LASER_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IR_CUT_PIN, GPIO.OUT, initial=GPIO.LOW)
    print("[GPIO] Initialized (Laser OFF, IR-CUT Night Mode)")

    try:
        main()
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user")
    finally:
        if camera:
            camera.release()
        GPIO.output(LASER_PIN, GPIO.LOW)
        GPIO.cleanup()
        print("[GPIO] Cleanup done")
