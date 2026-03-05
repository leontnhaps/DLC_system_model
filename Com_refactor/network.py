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

# Preview는 큐에 누적하지 않고 최신 프레임 1개만 유지
_preview_lock = threading.Lock()
_latest_preview = None


def set_latest_preview(data: bytes):
    """Store only the latest preview frame."""
    global _latest_preview
    with _preview_lock:
        _latest_preview = data


def pop_latest_preview():
    """Get and clear the latest preview frame."""
    global _latest_preview
    with _preview_lock:
        data = _latest_preview
        _latest_preview = None
    return data


def _recv_exact(sock: socket.socket, size: int):
    """Receive exactly `size` bytes. Return None only on clean EOF before any bytes."""
    buf = bytearray()
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            if not buf:
                return None
            raise ConnectionError(f"socket closed during recv_exact({size})")
        buf += chunk
    return bytes(buf)


class GuiCtrlClient(threading.Thread):
    """제어 소켓 - 명령 전송 + 이벤트 수신 (자동 재연결)"""
    
    def __init__(self, host, port, bus=None):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.sock = None
        self.bus = bus

    def _publish(self, tag, payload):
        if self.bus is not None:
            self.bus.publish(tag, payload)
        else:
            ui_q.put((tag, payload))
    
    def run(self):
        """자동 재연결 루프"""
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.connect((self.host, self.port))
                self.sock = s
                self._publish("toast", f"CTRL connected {self.host}:{self.port}")
                
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
                            self._publish("evt", evt)
                        except:
                            continue
            except Exception as e:
                self._publish("toast", f"CTRL err: {e}. Retry in 3s...")
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
    
    def __init__(self, host, port, outdir: pathlib.Path, bus=None):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.outdir = outdir
        self.sock = None
        self.bus = bus

    def _publish(self, tag, payload):
        if self.bus is not None:
            self.bus.publish(tag, payload)
        else:
            ui_q.put((tag, payload))
    
    def run(self):
        """자동 재연결 루프"""
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.host, self.port))
                self.sock = s
                self._publish("toast", f"IMG connected {self.host}:{self.port}")
                
                while True:
                    hdr = _recv_exact(s, 2)
                    if hdr is None:
                        break
                    (nlen,) = struct.unpack("<H", hdr)
                    name_bytes = _recv_exact(s, nlen)
                    if name_bytes is None:
                        raise ConnectionError("img closed during name recv")
                    name = name_bytes.decode("utf-8", "ignore")

                    dlen_buf = _recv_exact(s, 4)
                    if dlen_buf is None:
                        raise ConnectionError("img closed during length recv")
                    (dlen,) = struct.unpack("<I", dlen_buf)

                    data = _recv_exact(s, dlen)
                    if data is None:
                        raise ConnectionError("img closed during payload recv")
                    
                    # 프리뷰는 ui_q로
                    if name.startswith("_preview_"):
                        set_latest_preview(data)
                    else:
                        # 일반 이미지는 ui_q로만 전달 (저장은 event_handlers에서)
                        self._publish("saved", (name, data))
            except Exception as e:
                self._publish("toast", f"IMG err: {e}. Retry in 3s...")
                if self.sock:
                    try:
                        self.sock.close()
                    except:
                        pass
                    self.sock = None
                time.sleep(3)
