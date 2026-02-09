#!/usr/bin/env python3
"""
Network communication classes
Handles control and image channels with auto-reconnection
"""

import json
import socket
import struct
import threading
import pathlib
import time
import queue

# 전역 UI 큐 - 모든 이벤트가 여기로
ui_q: queue.Queue[tuple[str, object]] = queue.Queue()


class GuiCtrlClient(threading.Thread):
    """제어 소켓 - 명령 전송 + 이벤트 수신 (자동 재연결)"""
    
    def __init__(self, host, port):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.sock = None
    
    def run(self):
        """자동 재연결 루프"""
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.connect((self.host, self.port))
                self.sock = s
                ui_q.put(("toast", f"CTRL connected {self.host}:{self.port}"))
                
                # 이벤트 수신 루프
                buf = b""
                while True:
                    data = s.recv(4096)
                    if not data:
                        break
                    buf += data
                    
                    while True:
                        nl = buf.find(b"\n")
                        if nl < 0:
                            break
                        line = buf[:nl].decode("utf-8", "ignore").strip()
                        buf = buf[nl+1:]
                        if not line:
                            continue
                        try:
                            evt = json.loads(line)
                            ui_q.put(("evt", evt))
                        except:
                            continue
            except Exception as e:
                ui_q.put(("toast", f"CTRL err: {e}. Retry in 3s..."))
                if self.sock:
                    try:
                        self.sock.close()
                    except:
                        pass
                    self.sock = None
                time.sleep(3)
    
    def send(self, obj: dict):
        """JSON 명령 전송"""
        if not self.sock:
            return
        try:
            self.sock.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode())
        except Exception as e:
            print(f"[CTRL] Send error: {e}")


class GuiImgClient(threading.Thread):
    """이미지 소켓 - 이미지 수신 (자동 재연결)"""
    
    def __init__(self, host, port, outdir: pathlib.Path):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.outdir = outdir
        self.sock = None
    
    def run(self):
        """자동 재연결 루프"""
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.host, self.port))
                self.sock = s
                ui_q.put(("toast", f"IMG connected {self.host}:{self.port}"))
                
                while True:
                    hdr = s.recv(2)
                    if not hdr:
                        break
                    (nlen,) = struct.unpack("<H", hdr)
                    name = s.recv(nlen).decode("utf-8", "ignore")
                    (dlen,) = struct.unpack("<I", s.recv(4))
                    
                    buf = bytearray()
                    remain = dlen
                    while remain > 0:
                        chunk = s.recv(min(65536, remain))
                        if not chunk:
                            raise ConnectionError("img closed")
                        buf += chunk
                        remain -= len(chunk)
                    
                    data = bytes(buf)
                    
                    # 프리뷰는 ui_q로
                    if name.startswith("_preview_"):
                        ui_q.put(("preview", data))
                    else:
                        # 일반 이미지는 ui_q로만 전달 (저장은 event_handlers에서)
                        ui_q.put(("saved", (name, data)))
            except Exception as e:
                ui_q.put(("toast", f"IMG err: {e}. Retry in 3s..."))
                if self.sock:
                    try:
                        self.sock.close()
                    except:
                        pass
                    self.sock = None
                time.sleep(3)
