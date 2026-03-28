"""Preview frame UI component."""

from tkinter import Frame, Label
from PIL import Image, ImageTk, ImageDraw
import io


class PreviewFrame:
    """프리뷰 디스플레이 프레임"""
    def __init__(self, parent, width=640, height=480):
        self.width = width
        self.height = height
        self.overlay_roi = None
        self.overlay_roi_source_size = None
        self.overlay_roi_label = "LED ROI"

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
            src_default_w, src_default_h = img.size

            img.thumbnail((self.width, self.height), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(img)
            cx = img.width // 2
            cy = img.height // 2
            arm = max(8, min(img.width, img.height) // 30)

            # High-contrast crosshair: black outline + white inner lines.
            draw.line((cx - arm, cy, cx + arm, cy), fill="black", width=4)
            draw.line((cx, cy - arm, cx, cy + arm), fill="black", width=4)
            draw.line((cx - arm, cy, cx + arm, cy), fill="white", width=2)
            draw.line((cx, cy - arm, cx, cy + arm), fill="white", width=2)

            roi = self.overlay_roi
            if roi is not None:
                try:
                    rx, ry, rw, rh = [float(v) for v in roi]
                    if rw > 0 and rh > 0:
                        src_size = self.overlay_roi_source_size
                        if (
                            isinstance(src_size, (tuple, list))
                            and len(src_size) == 2
                            and float(src_size[0]) > 0
                            and float(src_size[1]) > 0
                        ):
                            src_w, src_h = float(src_size[0]), float(src_size[1])
                        else:
                            src_w, src_h = float(src_default_w), float(src_default_h)

                        sx = float(img.width) / max(1.0, src_w)
                        sy = float(img.height) / max(1.0, src_h)
                        x1 = max(0.0, min(float(img.width), rx * sx))
                        y1 = max(0.0, min(float(img.height), ry * sy))
                        x2 = max(0.0, min(float(img.width), (rx + rw) * sx))
                        y2 = max(0.0, min(float(img.height), (ry + rh) * sy))

                        if x2 > x1 and y2 > y1:
                            draw.rectangle((x1, y1, x2, y2), outline="#00FFFF", width=3)
                            label_y = max(0.0, y1 - 16.0)
                            draw.text((x1 + 2.0, label_y), self.overlay_roi_label or "LED ROI", fill="#00FFFF")
                except Exception:
                    pass

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

    def set_overlay_roi(self, roi=None, source_size=None, label="LED ROI"):
        """프리뷰 ROI 오버레이 갱신"""
        try:
            self.overlay_roi = None if roi is None else tuple(float(v) for v in roi)
        except Exception:
            self.overlay_roi = None
        try:
            if source_size is None:
                self.overlay_roi_source_size = None
            else:
                self.overlay_roi_source_size = (float(source_size[0]), float(source_size[1]))
        except Exception:
            self.overlay_roi_source_size = None
        self.overlay_roi_label = str(label or "LED ROI")
