#!/usr/bin/env python3
"""
Com Client - Step 1: 기본 구조
나중에 프리뷰 기능 추가 예정
"""

from tkinter import Tk, Label, Button, Frame
from PIL import Image, ImageTk

class App:
    def __init__(self, root: Tk):
        self.root = root
        root.title("IR Camera Test")
        root.geometry("800x600")
        
        # Top Bar
        top = Frame(root)
        top.pack(fill="x", padx=10, pady=6)
        Label(top, text="🎥 Step 1: Camera Capture Test").pack(side="left")
        
        # Preview Box
        center = Frame(root)
        center.pack(fill="x", padx=10)
        self.PREV_W, self.PREV_H = 640, 480
        self.preview_box = Frame(center, width=self.PREV_W, height=self.PREV_H,
                                 bg="#111", highlightthickness=1, highlightbackground="#333")
        self.preview_box.pack()
        self.preview_box.pack_propagate(False)
        self.preview_label = Label(self.preview_box, bg="#111", 
                                   text="연결 대기 중...", fg="white")
        self.preview_label.place(x=0, y=0, width=self.PREV_W, height=self.PREV_H)
        
        # Info
        info = Frame(root)
        info.pack(fill="x", padx=10, pady=10)
        Label(info, text="Step 1: 라즈베리파이에서 capture_test.py 실행 후 test_capture.jpg 확인").pack()
    
    def run(self):
        self.root.mainloop()

def main():
    root = Tk()
    App(root).run()

if __name__ == "__main__":
    main()
