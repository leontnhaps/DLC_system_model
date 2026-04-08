"""LED Test tab UI component."""

from tkinter import Label, Button, Frame, StringVar, IntVar, Canvas, Scrollbar
from tkinter import ttk
from PIL import Image, ImageTk
import cv2


class LEDTestTab:
    """LED Test 탭 UI."""

    def __init__(self, parent, callbacks, initial_params=None):
        self.callbacks = callbacks
        self.initial_params = dict(initial_params or {})

        canvas = Canvas(parent)
        scrollbar = Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.frame = Frame(canvas)

        self.frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._build()

    def _build(self):
        p = self.initial_params
        self.r_min = IntVar(value=int(p.get("r_min", 60)))
        self.g_min = IntVar(value=int(p.get("g_min", 60)))
        self.b_min = IntVar(value=int(p.get("b_min", 60)))
        self.rg_min = IntVar(value=int(p.get("rg_min", 10)))
        self.rb_min = IntVar(value=int(p.get("rb_min", 100)))
        self.gr_min = IntVar(value=int(p.get("gr_min", 10)))
        self.gb_min = IntVar(value=int(p.get("gb_min", 10)))
        self.br_min = IntVar(value=int(p.get("br_min", 40)))
        self.bg_min = IntVar(value=int(p.get("bg_min", 40)))
        self.min_pixels = IntVar(value=int(p.get("min_pixels", 50)))

        self.status_text = StringVar(value="대기 중")
        self.raw_r_text = StringVar(value="0")
        self.raw_b_text = StringVar(value="0")
        self.raw_g_text = StringVar(value="0")
        self.threshold_text = StringVar(value=str(int(self.min_pixels.get())))
        self.bits_text = StringVar(value="000")
        self.roi_text = StringVar(value="-")
        self.legacy_text = StringVar(value="NONE")

        row = 0
        Label(self.frame, text="LED Test", font=("", 12, "bold")).grid(
            row=row, column=0, columnspan=4, sticky="w", padx=5, pady=(5, 10)
        )
        row += 1

        btn_frame = Frame(self.frame)
        btn_frame.grid(row=row, column=0, columnspan=4, sticky="w", padx=5, pady=(0, 8))
        self.btn_start = Button(
            btn_frame,
            text="Start",
            command=self._on_start,
            width=12,
            bg="#388E3C",
            fg="white",
            font=("", 10, "bold"),
        )
        self.btn_start.pack(side="left", padx=(0, 6))
        self.btn_stop = Button(
            btn_frame,
            text="Stop",
            command=self._on_stop,
            width=12,
            bg="#D32F2F",
            fg="white",
            font=("", 10, "bold"),
            state="disabled",
        )
        self.btn_stop.pack(side="left")
        row += 1

        self.status_label = Label(self.frame, textvariable=self.status_text, fg="#333", font=("", 10))
        self.status_label.grid(row=row, column=0, columnspan=4, sticky="w", padx=5, pady=(0, 8))
        row += 1

        Label(self.frame, text="ROI Preview", font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=4, sticky="w", padx=5, pady=(0, 4)
        )
        row += 1

        preview_frame = Frame(
            self.frame,
            width=420,
            height=300,
            bg="#111",
            highlightthickness=1,
            highlightbackground="#333",
        )
        preview_frame.grid(row=row, column=0, columnspan=4, sticky="w", padx=5, pady=(0, 8))
        preview_frame.pack_propagate(False)
        self.preview_label = Label(preview_frame, bg="#111", fg="#666", text="(Waiting for LED Test...)")
        self.preview_label.pack(fill="both", expand=True)
        row += 1

        result_frame = Frame(self.frame)
        result_frame.grid(row=row, column=0, columnspan=4, sticky="we", padx=5, pady=(0, 8))
        row += 1

        Label(result_frame, text="R raw").grid(row=0, column=0, sticky="w", padx=(0, 6))
        Label(result_frame, textvariable=self.raw_r_text, fg="#D32F2F", font=("", 10, "bold")).grid(row=0, column=1, sticky="w", padx=(0, 12))
        Label(result_frame, text="B raw").grid(row=0, column=2, sticky="w", padx=(0, 6))
        Label(result_frame, textvariable=self.raw_b_text, fg="#1976D2", font=("", 10, "bold")).grid(row=0, column=3, sticky="w", padx=(0, 12))
        Label(result_frame, text="G raw").grid(row=0, column=4, sticky="w", padx=(0, 6))
        Label(result_frame, textvariable=self.raw_g_text, fg="#388E3C", font=("", 10, "bold")).grid(row=0, column=5, sticky="w")

        Label(result_frame, text="Threshold").grid(row=1, column=0, sticky="w", padx=(0, 6))
        Label(result_frame, textvariable=self.threshold_text, font=("", 10, "bold")).grid(row=1, column=1, sticky="w", padx=(0, 12))
        Label(result_frame, text="3bit").grid(row=1, column=2, sticky="w", padx=(0, 6))
        self.bits_value_label = Label(
            result_frame,
            textvariable=self.bits_text,
            font=("Consolas", 26, "bold"),
            bg="#111111",
            fg="#FFD54F",
            padx=18,
            pady=6,
            relief="solid",
            bd=1,
        )
        self.bits_value_label.grid(row=1, column=3, sticky="w", padx=(0, 12))
        Label(result_frame, text="Legacy").grid(row=1, column=4, sticky="w", padx=(0, 6))
        Label(result_frame, textvariable=self.legacy_text, font=("", 10, "bold")).grid(row=1, column=5, sticky="w")
        Label(result_frame, text="R   B   G", fg="#666", font=("Consolas", 9, "bold")).grid(row=2, column=3, sticky="w", padx=(0, 12))

        Label(result_frame, text="ROI").grid(row=3, column=0, sticky="w", padx=(0, 6))
        Label(result_frame, textvariable=self.roi_text, fg="#555").grid(row=3, column=1, columnspan=5, sticky="w")

        Label(self.frame, text="Filter Params", font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=4, sticky="w", padx=5, pady=(0, 4)
        )
        row += 1

        self._entry(row, "R Min", self.r_min); row += 1
        self._entry(row, "G Min", self.g_min); row += 1
        self._entry(row, "B Min", self.b_min); row += 1
        self._entry(row, "R-G Min", self.rg_min); row += 1
        self._entry(row, "R-B Min", self.rb_min); row += 1
        self._entry(row, "G-R Min", self.gr_min); row += 1
        self._entry(row, "G-B Min", self.gb_min); row += 1
        self._entry(row, "B-R Min", self.br_min); row += 1
        self._entry(row, "B-G Min", self.bg_min); row += 1
        self._entry(row, "Threshold / Min Pixels", self.min_pixels); row += 1

        for c in range(4):
            self.frame.grid_columnconfigure(c, weight=1)

    def _entry(self, row, text, var):
        Label(self.frame, text=text).grid(row=row, column=0, sticky="w", padx=(5, 10), pady=2)
        ttk.Entry(self.frame, textvariable=var, width=12).grid(row=row, column=1, sticky="w", padx=2, pady=2)

    def _on_start(self):
        if self.callbacks.get("start_led_test"):
            self.callbacks["start_led_test"]()

    def _on_stop(self):
        if self.callbacks.get("stop_led_test"):
            self.callbacks["stop_led_test"]()

    def get_filter_params(self):
        return {
            "r_min": max(0, int(self.r_min.get())),
            "g_min": max(0, int(self.g_min.get())),
            "b_min": max(0, int(self.b_min.get())),
            "rg_min": int(self.rg_min.get()),
            "rb_min": int(self.rb_min.get()),
            "gr_min": int(self.gr_min.get()),
            "gb_min": int(self.gb_min.get()),
            "br_min": int(self.br_min.get()),
            "bg_min": int(self.bg_min.get()),
            "min_pixels": max(0, int(self.min_pixels.get())),
        }

    def set_running_state(self, is_running):
        self.btn_start.config(state="disabled" if is_running else "normal")
        self.btn_stop.config(state="normal" if is_running else "disabled")

    def update_status(self, text, fg="#333"):
        self.status_text.set(str(text))
        self.status_label.config(fg=fg)

    def update_result(self, raw_score=None, bits="000", threshold=None, roi=None, legacy_pred="NONE"):
        raw_score = dict(raw_score or {})
        self.raw_r_text.set(str(int(raw_score.get("R", 0))))
        self.raw_b_text.set(str(int(raw_score.get("B", 0))))
        self.raw_g_text.set(str(int(raw_score.get("G", 0))))
        if threshold is None:
            threshold = self.min_pixels.get()
        self.threshold_text.set(str(int(threshold)))
        bits_text = str(bits or "000")
        self.bits_text.set(bits_text)
        self.legacy_text.set(str(legacy_pred or "NONE"))
        if hasattr(self, "bits_value_label"):
            active_bits = sum(1 for ch in bits_text if ch == "1")
            if active_bits >= 2:
                fg = "#FFF176"
                bg = "#1B1B1B"
            elif active_bits == 1:
                fg = "#FFE082"
                bg = "#1B1B1B"
            else:
                fg = "#BDBDBD"
                bg = "#1B1B1B"
            self.bits_value_label.config(fg=fg, bg=bg)
        if roi is None:
            self.roi_text.set("-")
        else:
            try:
                x, y, w, h = [int(round(float(v))) for v in roi]
                self.roi_text.set(f"x={x} y={y} w={w} h={h}")
            except Exception:
                self.roi_text.set(str(roi))

    def show_preview(self, img_bgr):
        if img_bgr is None:
            self.preview_label.config(text="(No ROI preview)", image="")
            self.preview_label.image = None
            return
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        im.thumbnail((420, 300), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(im)
        self.preview_label.config(image=photo, text="")
        self.preview_label.image = photo
