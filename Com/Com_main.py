#!/usr/bin/env python3
"""
Com Client - Modularized version
"""

import queue
from tkinter import Tk, Label, Frame
from network import GuiCtrlClient, GuiImgClient
from ui_components import PreviewFrame, ControlPanel

SERVER_HOST = "127.0.0.1"
GUI_CTRL_PORT = 7600
GUI_IMG_PORT = 7601

class App:
    def __init__(self, root: Tk):
        self.root = root
        root.title("IR Camera System")
        root.geometry("800x650")
        
        # 이미지 큐
        self.img_queue = queue.Queue()
        
        # Top Bar
        top = Frame(root)
        top.pack(fill="x", padx=10, pady=6)
        Label(top, text="🎥 IR-CUT Camera Control").pack(side="left")
        
        # Preview Frame
        center = Frame(root)
        center.pack(fill="x", padx=10)
        self.preview_frame = PreviewFrame(center, width=640, height=480)
        
        # Control Panel
        callbacks = {
            'toggle_preview': self.toggle_preview,
            'start_preview': self.start_preview,
            'set_ir_cut': self.set_ir_cut
        }
        self.control_panel = ControlPanel(root, callbacks)
        
        # Info Label
        info = Frame(root)
        info.pack(fill="x", padx=10, pady=10)
        self.info_label = Label(info, text="라즈베리파이 에이전트 연결 대기 중...")
        self.info_label.pack()
        
        # Network Clients
        self.ctrl_client = GuiCtrlClient(SERVER_HOST, GUI_CTRL_PORT)
        self.ctrl_client.start()
        
        self.img_client = GuiImgClient(SERVER_HOST, GUI_IMG_PORT, self.img_queue)
        self.img_client.start()
        
        # Polling
        self.frame_count = 0
        self.root.after(50, self._poll)
    
    def toggle_preview(self, enable):
        """프리뷰 토글"""
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
        """프리뷰 시작 (체크박스 자동 체크)"""
        self.control_panel.preview_enable.set(True)
        self.toggle_preview(True)
    
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
            tag, payload = self.img_queue.get_nowait()
            if tag == "img":
                self.preview_frame.display_image(payload)
                self.frame_count += 1
                self.info_label.config(text=f"✅ 프레임 수신 중... ({self.frame_count})")
        except queue.Empty:
            pass
        
        self.root.after(50, self._poll)
    
    def run(self):
        self.root.mainloop()

def main():
    root = Tk()
    App(root).run()

if __name__ == "__main__":
    main()
