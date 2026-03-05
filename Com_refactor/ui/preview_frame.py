"""Preview frame UI component."""

from tkinter import Frame, Label
from PIL import Image, ImageTk, ImageDraw
import io


class PreviewFrame:
    """프리뷰 디스플레이 프레임"""
    def __init__(self, parent, width=640, height=480):
        self.width = width
        self.height = height

        self.frame = Frame(parent, width=width, height=height,
                          bg="#111", highlightthickness=1, highlightbackground="#333")
        self.frame.pack()
        self.frame.pack_propagate(False)

        self.label = Label(self.frame, bg="#111")
        self.label.place(x=0, y=0, width=width, height=height)

        # Overlay (hotkey/current ID)
        self.overlay_label = Label(
            self.frame,
            text="Shoot Dwell: - | ID: - | Idle",
            bg="#000000",
            fg="#FFFFFF",
            font=("", 9, "bold"),
            padx=6,
            pady=3,
        )
        self.overlay_label.place(x=8, y=8)

    def display_image(self, jpeg_bytes):
        """이미지 표시"""
        try:
            img = Image.open(io.BytesIO(jpeg_bytes))

            img.thumbnail((self.width, self.height), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            self.label.config(image=tk_img)
            self.label.image = tk_img
        except Exception as e:
            print(f"[DISPLAY] 오류: {e}")

    def set_overlay_text(self, text):
        """프리뷰 오버레이 텍스트 갱신"""
        try:
            self.overlay_label.config(text=text)
        except Exception:
            pass
