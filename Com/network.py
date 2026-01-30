#!/usr/bin/env python3
"""
Network clients for server communication
"""

import json, socket, struct, threading

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
    def __init__(self, host, port, img_queue):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.sock = None
        self.img_queue = img_queue
        
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
                self.img_queue.put(("img", name, data))
                
        except Exception as e:
            print(f"[IMG] 오류: {e}")
