import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button, Scale, HORIZONTAL, DoubleVar
import cv2
import numpy as np
import os

class UndistortApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Undistort Image Tool")
        self.root.geometry("400x350")
        
        # Calibration Data
        self.calib_path = None
        self.K = None
        self.D = None
        self.model = None
        self.img_size = None
        
        # UI Elements
        Label(root, text="1. Load Calibration File (.npz)", font=("Arial", 10, "bold")).pack(pady=(10, 5))
        self.btn_load_calib = Button(root, text="Load calib.npz", command=self.load_calib)
        self.btn_load_calib.pack()
        self.lbl_calib_path = Label(root, text="No file loaded", fg="gray", wraplength=380)
        self.lbl_calib_path.pack(pady=5)
        
        Label(root, text="2. Settings", font=("Arial", 10, "bold")).pack(pady=(10, 5))
        self.alpha = DoubleVar(value=0.0)
        Label(root, text="Alpha / Balance (0.0 ~ 1.0)").pack()
        Scale(root, from_=0.0, to=1.0, resolution=0.01, orient=HORIZONTAL, variable=self.alpha, length=300).pack()
        
        Label(root, text="3. Select Image & Convert", font=("Arial", 10, "bold")).pack(pady=(15, 5))
        self.btn_select_img = Button(root, text="Select Image (.jpg/.png)", command=self.process_image, state="disabled")
        self.btn_select_img.pack()
        
        self.lbl_status = Label(root, text="Ready", fg="blue")
        self.lbl_status.pack(pady=20)

    def load_calib(self):
        path = filedialog.askopenfilename(filetypes=[("NPZ Files", "*.npz")])
        if not path: return
        
        try:
            cal = np.load(path, allow_pickle=True)
            self.model = str(cal["model"])
            self.K = cal["K"].astype(np.float32)
            self.D = cal["D"].astype(np.float32)
            self.img_size = tuple(int(x) for x in cal["img_size"])
            
            self.calib_path = path
            self.lbl_calib_path.config(text=os.path.basename(path), fg="green")
            self.btn_select_img.config(state="normal")
            self.lbl_status.config(text=f"Loaded: {self.model}, {self.img_size}", fg="blue")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration:\n{e}")
            self.lbl_calib_path.config(text="Load failed", fg="red")

    def _scale_K(self, K, sx, sy):
        K2 = K.copy()
        K2[0,0]*=sx; K2[1,1]*=sy
        K2[0,2]*=sx; K2[1,2]*=sy
        K2[2,2]=1.0
        return K2

    def process_image(self):
        if self.K is None: return
        
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
        if not path: return
        
        try:
            # 1. Load Image
            # Use cv2.imdecode for unicode path support
            nparr = np.fromfile(path, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to read image")
            
            h, w = img.shape[:2]
            
            # 2. Compute Maps
            Wc, Hc = self.img_size
            sx, sy = w/float(Wc), h/float(Hc)
            K = self._scale_K(self.K, sx, sy)
            D = self.D
            alpha = self.alpha.get()
            
            if self.model == "pinhole":
                newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w,h), alpha=alpha, newImgSize=(w,h))
                m1, m2 = cv2.initUndistortRectifyMap(K, D, None, newK, (w,h), cv2.CV_16SC2)
            else: # fisheye
                R = np.eye(3, dtype=np.float32)
                newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K, D, (w,h), R, balance=alpha, new_size=(w,h)
                )
                m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, newK, (w,h), cv2.CV_16SC2)
            
            # 3. Remap
            ud_img = cv2.remap(img, m1, m2, cv2.INTER_LINEAR)
            
            # 4. Save
            base, ext = os.path.splitext(path)
            out_path = f"{base}.ud{ext}"
            
            # Use cv2.imencode for unicode path support
            ret, buf = cv2.imencode(ext, ud_img)
            if ret:
                with open(out_path, "wb") as f:
                    buf.tofile(f)
                self.lbl_status.config(text=f"Saved: {os.path.basename(out_path)}", fg="green")
                messagebox.showinfo("Success", f"Saved undistorted image:\n{out_path}")
            else:
                raise ValueError("Failed to encode image")
                
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed:\n{e}")
            self.lbl_status.config(text="Error", fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = UndistortApp(root)
    root.mainloop()
