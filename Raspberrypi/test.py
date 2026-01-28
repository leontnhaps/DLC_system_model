#!/usr/bin/env python3
"""
Step 4: IR-CUT 필터 제어 추가
- GPIO 17: IR-CUT 제어
- Day Mode / Night Mode 전환
"""
import socket, struct, io, time, json, threading
from picamera2 import Picamera2
import RPi.GPIO as GPIO
# ==================== GPIO 설정 ====================
IR_CUT_PIN = 17  # BCM 17번 (물리 11번 핀)
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
            jpeg_data = bio.getvalue()
            
            push_image(img_sock, f"_preview_{int(time.time()*1000)}.jpg", jpeg_data)
            time.sleep(interval)
            
        print("[PREVIEW] 중지됨")
    except Exception as e:
        print(f"[PREVIEW] 오류: {e}")
# ==================== 메인 ====================
def main():
    global picam, preview_thread
    
    # GPIO 초기화
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(IR_CUT_PIN, GPIO.OUT, initial=GPIO.LOW)
    print("[GPIO] IR-CUT 초기화: Night Mode (GPIO 17)")
    
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
    print("Step 4: 명령 대기 중...")
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
                
                elif c == "ir_cut":
                    # IR-CUT 제어
                    mode = cmd.get("mode", "day")  # "day" or "night"
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