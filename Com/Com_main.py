#!/usr/bin/env python3
"""
Com Client - Step 3: 프리뷰 스트리밍
"""

import json, socket, struct, threading, queue
from tkinter import Tk, Label, Button, Frame, BooleanVar, Checkbutton
from PIL import Image, ImageTk
import io

SERVER_HOST = "127.0.0.1"
GUI_CTRL_PORT = 7600
GUI_IMG_PORT = 7601

# 큐
img_queue = queue.Queue()

class GuiCtrlClient(threading.Thread):
    """제어 소켓 - 명령 전송"""
    def __init__(self, host, port):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.sock = None
        
    def run(self):
        print(f"[CTRL] 연결 중: {self.host}:{self.port}")
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            print(f"[CTRL] 연결 성공!")
        except Exception as e:
            print(f"[CTRL] 오류: {e}")
    
    def send(self, cmd: dict):
        """명령 전송"""
        if self.sock:
            try:
                data = (json.dumps(cmd) + "\n").encode()
                self.sock.sendall(data)
                print(f"[CTRL] 전송: {cmd.get('cmd', '?')}")
            except Exception as e:
                print(f"[CTRL] 전송 오류: {e}")

class GuiImgClient(threading.Thread):
    """이미지 소켓 - 이미지 수신"""
    def __init__(self, host, port):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.sock = None
        
    def run(self):
        print(f"[IMG] 연결 중: {self.host}:{self.port}")
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            print(f"[IMG] 연결 성공!")
            
            while True:
                # 이미지 수신
                hdr = self.sock.recv(2)
                if not hdr: break
                
                (name_len,) = struct.unpack("<H", hdr)
                name = self.sock.recv(name_len).decode("utf-8")
                (dlen,) = struct.unpack("<I", self.sock.recv(4))
                
                buf = bytearray()
                remain = dlen
                while remain > 0:
                    chunk = self.sock.recv(min(65536, remain))
                    if not chunk: raise ConnectionError()
                    buf += chunk
                    remain -= len(chunk)
                
                data = bytes(buf)
                img_queue.put(("img", data))
                
        except Exception as e:
            print(f"[IMG] 오류: {e}")

class App:
    def __init__(self, root: Tk):
        self.root = root
        root.title("IR Camera - Step 3")
        root.geometry("800x650")
        
        # Top Bar
        top = Frame(root)
        top.pack(fill="x", padx=10, pady=6)
        Label(top, text="🎥 Step 3: Preview Streaming").pack(side="left")
        
        # Preview Box
        center = Frame(root)
        center.pack(fill="x", padx=10)
        self.PREV_W, self.PREV_H = 640, 480
        self.preview_box = Frame(center, width=self.PREV_W, height=self.PREV_H,
                                 bg="#111", highlightthickness=1, highlightbackground="#333")
        self.preview_box.pack()
        self.preview_box.pack_propagate(False)
        self.preview_label = Label(self.preview_box, bg="#111")
        self.preview_label.place(x=0, y=0, width=self.PREV_W, height=self.PREV_H)
        
        # Controls
        controls = Frame(root)
        controls.pack(fill="x", padx=10, pady=10)
        
        self.preview_enable = BooleanVar(value=False)
        Checkbutton(controls, text="프리뷰 활성화", variable=self.preview_enable,
                   command=self.toggle_preview).pack(side="left", padx=5)
        
        Button(controls, text="프리뷰 시작", command=lambda: self.start_preview()).pack(side="left", padx=5)
        
        # IR-CUT Controls
        ir_frame = Frame(controls, relief="ridge", borderwidth=1)
        ir_frame.pack(side="left", padx=10)
        Label(ir_frame, text="IR-CUT:").pack(side="left", padx=3)
        Button(ir_frame, text="☀️ Day Mode", command=lambda: self.set_ir_cut("day"),
               bg="#FFE0B2").pack(side="left", padx=2)
        Button(ir_frame, text="🌙 Night Mode", command=lambda: self.set_ir_cut("night"),
               bg="#444", fg="white").pack(side="left", padx=2)
        
        # Info
        info = Frame(root)
        info.pack(fill="x", padx=10, pady=10)
        self.info_label = Label(info, text="라즈베리파이에서 agent_step3.py 실행 대기 중...")
        self.info_label.pack()
        
        # 클라이언트 시작
        self.ctrl_client = GuiCtrlClient(SERVER_HOST, GUI_CTRL_PORT)
        self.ctrl_client.start()
        
        self.img_client = GuiImgClient(SERVER_HOST, GUI_IMG_PORT)
        self.img_client.start()
        
        # 폴링
        self.frame_count = 0
        self.root.after(100, self._poll)
    
    def toggle_preview(self):
        """프리뷰 토글"""
        enable = self.preview_enable.get()
        cmd = {
            "cmd": "preview",
            "enable": enable,
            "width": 640,
            "height": 480,
            "fps": 5,
            "quality": 70
        }
        self.ctrl_client.send(cmd)
        
        if enable:
            self.info_label.config(text="✅ 프리뷰 활성화됨")
        else:
            self.info_label.config(text="⏸️ 프리뷰 중지됨")
    
    def start_preview(self):
        """프리뷰 시작 (체크박스 상관없이)"""
        self.preview_enable.set(True)
        self.toggle_preview()
    
    def set_ir_cut(self, mode):
        """IR-CUT 모드 설정"""
        cmd = {
            "cmd": "ir_cut",
            "mode": mode
        }
        self.ctrl_client.send(cmd)
        
        if mode == "day":
            self.info_label.config(text="☀️ Day Mode (IR 필터 ON)")
        else:
            self.info_label.config(text="🌙 Night Mode (IR 필터 OFF)")
    
    def _poll(self):
        """이미지 수신 체크"""
        try:
            tag, payload = img_queue.get_nowait()
            if tag == "img":
                self._display_image(payload)
                self.frame_count += 1
                self.info_label.config(text=f"✅ 프레임 수신 중... ({self.frame_count})")
        except queue.Empty:
            pass
        
        self.root.after(50, self._poll)
    
    def _display_image(self, jpeg_bytes):
        """이미지 표시"""
        try:
            img = Image.open(io.BytesIO(jpeg_bytes))
            img.thumbnail((self.PREV_W, self.PREV_H), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            self.preview_label.config(image=tk_img)
            self.preview_label.image = tk_img
        except Exception as e:
            print(f"[DISPLAY] 오류: {e}")
    
    def run(self):
        self.root.mainloop()

def main():
    root = Tk()
    App(root).run()

if __name__ == "__main__":
    main()
