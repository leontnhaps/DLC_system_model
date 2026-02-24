#!/usr/bin/env python3
# pc_server.py — Headless broker between Raspberry Pi agent and GUI client
import json, socket, socketserver, struct, threading, pathlib, time
from datetime import datetime

# Ports
AGENT_CTRL_PORT = 7500   # Pi agent connects here (control JSONL)
AGENT_IMG_PORT  = 7501   # Pi agent connects here (image frames)
GUI_CTRL_PORT   = 7600   # GUI connects here (control JSONL)
GUI_IMG_PORT    = 7601   # GUI connects here (image frames)

# Optional server-side saving (GUI도 저장하므로 기본 False 권장)
SAVE_ON_SERVER = False
OUT_DIR = pathlib.Path(f"captures_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
if SAVE_ON_SERVER:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= Shared state =========
agent_ctrl_lock = threading.Lock()
agent_ctrl_sock = None           # single agent (one Pi)

gui_ctrl_lock = threading.Lock()
gui_ctrl_clients = set()         # set[socket.socket]

gui_img_lock = threading.Lock()
gui_img_clients = set()          # set[socket.socket]

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

def _send_line(sock, obj):
    data = (json.dumps(obj, separators=(",",":"))+"\n").encode()
    sock.sendall(data)

def broadcast_event(evt: dict):
    """Send JSON event to all GUI control clients."""
    dead = []
    with gui_ctrl_lock:
        for s in list(gui_ctrl_clients):
            try:
                _send_line(s, evt)
            except Exception:
                dead.append(s)
        for s in dead:
            gui_ctrl_clients.discard(s)
            try: s.close()
            except: pass

def forward_image_to_guis(name: str, data: bytes):
    """Frame: [uint16 name_len][name][uint32 data_len][data]"""
    frame = struct.pack("<H", len(name.encode())) + name.encode() + struct.pack("<I", len(data))
    dead = []
    with gui_img_lock:
        for s in list(gui_img_clients):
            try:
                s.sendall(frame); s.sendall(data)
            except Exception:
                dead.append(s)
        for s in dead:
            gui_img_clients.discard(s)
            try: s.close()
            except: pass

# ========= Agent (Pi) handlers =========
class AgentControlHandler(socketserver.BaseRequestHandler):
    def handle(self):
        global agent_ctrl_sock
        sock = self.request
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Register
        with agent_ctrl_lock:
            agent_ctrl_sock = sock
        broadcast_event({"event":"agent","state":"connected","addr":self.client_address})

        buf = b""
        try:
            while True:
                data = sock.recv(4096)
                if not data: break
                buf += data
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0: break
                    line = buf[:nl].decode("utf-8","ignore").strip()
                    buf = buf[nl+1:]
                    if not line: continue
                    try:
                        evt = json.loads(line)
                    except Exception:
                        broadcast_event({"event":"error","message":"agent event parse error"})
                        continue
                    # fan-out to GUI clients
                    broadcast_event(evt)
        finally:
            with agent_ctrl_lock:
                if agent_ctrl_sock is sock:
                    agent_ctrl_sock = None
            broadcast_event({"event":"agent","state":"disconnected","addr":self.client_address})

class AgentImageHandler(socketserver.BaseRequestHandler):
    def handle(self):
        sock = self.request
        try:
            while True:
                hdr = _recv_exact(sock, 2)
                if hdr is None:
                    break
                (name_len,) = struct.unpack("<H", hdr)
                name_bytes = _recv_exact(sock, name_len)
                if name_bytes is None:
                    raise ConnectionError("agent image socket closed during name recv")
                name = name_bytes.decode("utf-8","ignore")

                dlen_buf = _recv_exact(sock, 4)
                if dlen_buf is None:
                    raise ConnectionError("agent image socket closed during length recv")
                (dlen,) = struct.unpack("<I", dlen_buf)

                data = _recv_exact(sock, dlen)
                if data is None:
                    raise ConnectionError("agent image socket closed during payload recv")

                # optional save on server
                if SAVE_ON_SERVER and not name.startswith("_preview_"):
                    OUT_DIR.mkdir(parents=True, exist_ok=True)
                    with open(OUT_DIR / name, "wb") as f:
                        f.write(data)

                # forward to all GUI image clients
                forward_image_to_guis(name, data)
        except Exception as e:
            broadcast_event({"event":"warn","message":f"agent image err: {e}"})

class AgentControlServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
class AgentImageServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

# ========= GUI handlers =========
class GuiControlHandler(socketserver.BaseRequestHandler):
    def handle(self):
        sock = self.request
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with gui_ctrl_lock:
            gui_ctrl_clients.add(sock)
        # greet + current agent state
        with agent_ctrl_lock:
            state = "connected" if agent_ctrl_sock else "disconnected"
        try:
            _send_line(sock, {"event":"hello","agent_state":state})
        except: pass

        buf = b""
        try:
            while True:
                data = sock.recv(4096)
                if not data: break
                buf += data
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0: break
                    line = buf[:nl].decode("utf-8","ignore").strip()
                    buf = buf[nl+1:]
                    if not line: continue
                    # GUI command → forward to agent (if any)
                    try:
                        cmd = json.loads(line)
                    except Exception:
                        _send_line(sock, {"event":"error","message":"bad json"}); continue

                    with agent_ctrl_lock:
                        agent = agent_ctrl_sock
                    if agent is None:
                        _send_line(sock, {"event":"error","message":"no agent connected"})
                        continue

                    try:
                        _send_line(agent, cmd)  # forward as-is
                        # Optionally ack GUI
                        _send_line(sock, {"event":"ack","cmd":cmd.get("cmd","")})
                    except Exception as e:
                        _send_line(sock, {"event":"error","message":f"forward failed: {e}"})
        finally:
            with gui_ctrl_lock:
                if sock in gui_ctrl_clients:
                    gui_ctrl_clients.remove(sock)
            try: sock.close()
            except: pass

class GuiImageHandler(socketserver.BaseRequestHandler):
    def handle(self):
        sock = self.request
        sock.settimeout(2.0)  # Prevent blocking sendall from hanging the server
        with gui_img_lock:
            gui_img_clients.add(sock)
        try:
            while True:
                try:
                    if not sock.recv(1):
                        break
                except socket.timeout:
                    # Timeout on recv is expected since client doesn't send data.
                    # This allows the loop to continue and keeps the socket alive
                    # while enforcing timeout on send() operations elsewhere.
                    continue
        except Exception:
            pass
        finally:
            with gui_img_lock:
                gui_img_clients.discard(sock)
            try: sock.close()
            except: pass

class GuiControlServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
class GuiImageServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

def main():
    srvA = AgentControlServer(("0.0.0.0", AGENT_CTRL_PORT), AgentControlHandler)
    srvB = AgentImageServer (("0.0.0.0", AGENT_IMG_PORT),  AgentImageHandler)
    srvC = GuiControlServer  (("0.0.0.0", GUI_CTRL_PORT),   GuiControlHandler)
    srvD = GuiImageServer    (("0.0.0.0", GUI_IMG_PORT),    GuiImageHandler)

    threading.Thread(target=srvA.serve_forever, daemon=True).start()
    threading.Thread(target=srvB.serve_forever, daemon=True).start()
    threading.Thread(target=srvC.serve_forever, daemon=True).start()
    threading.Thread(target=srvD.serve_forever, daemon=True).start()

    print(f"[SERVER] Agent CTRL:{AGENT_CTRL_PORT} IMG:{AGENT_IMG_PORT} | GUI CTRL:{GUI_CTRL_PORT} IMG:{GUI_IMG_PORT}")
    if SAVE_ON_SERVER:
        print(f"[SERVER] Saving images to: {OUT_DIR.resolve()}")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        srvA.shutdown(); srvB.shutdown(); srvC.shutdown(); srvD.shutdown()

if __name__ == "__main__":
    main()
