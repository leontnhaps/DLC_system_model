#!/usr/bin/env python3
"""
Network communication classes for GUI client
Handles control and image channels to pc_server.py
"""

import json
import socket
import struct
import threading
import pathlib
import time
import queue

# Global UI queue (imported by main)
ui_q: "queue.Queue[tuple[str,object]]" = queue.Queue()


class GuiCtrlClient(threading.Thread):
    """Control channel - JSON commands and events"""
    
    def __init__(self, host, port):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.sock = None
    
    def run(self):
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # s.settimeout(10.0) # Optional: 타임아웃 설정
                s.connect((self.host, self.port))
                self.sock = s
                ui_q.put(("toast", f"CTRL connected {self.host}:{self.port}"))
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
                        except:
                            continue
                        ui_q.put(("evt", evt))
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
        """Send JSON command to server"""
        if not self.sock:
            return
        try:
            self.sock.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode())
        except Exception as e:
            print(f"[GuiCtrlClient] Send error: {e}")


class GuiImgClient(threading.Thread):
    """Image channel - receives binary image data"""
    
    def __init__(self, host, port, outdir: pathlib.Path):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.outdir = outdir
        self.sock = None
    
    def run(self):
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # s.settimeout(10.0)
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
                    if name.startswith("_preview_"):
                        ui_q.put(("preview", data))
                    else:
                        self.outdir.mkdir(parents=True, exist_ok=True)
                        with open(self.outdir / name, "wb") as f:
                            f.write(data)
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
