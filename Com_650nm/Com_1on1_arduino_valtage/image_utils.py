#!/usr/bin/env python3
"""
Image processing utilities
Handles calibration loading, undistortion (CPU/CUDA/Torch), and image loading
"""

import numpy as np
import cv2
import pathlib

# Optional PyTorch support
try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except Exception:
    torch = None
    F = None
    _TORCH_AVAILABLE = False


class ImageProcessor:
    """Handles image loading and undistortion with CUDA/Torch acceleration"""
    
    def __init__(self):
        # Calibration data
        self._ud_model = None
        self._ud_K = None
        self._ud_D = None
        self._ud_img_size = None
        self._ud_src_size = None
        
        # CPU undistortion maps
        self._ud_m1 = None
        self._ud_m2 = None
        
        # CUDA support
        self._use_cv2_cuda = False
        try:
            self._use_cv2_cuda = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            self._use_cv2_cuda = False
        self._ud_gm1 = None
        self._ud_gm2 = None
        
        # Torch support
        self._torch_available = _TORCH_AVAILABLE
        self._torch_cuda = bool(_TORCH_AVAILABLE and torch.cuda.is_available())
        self._torch_device = torch.device("cuda") if self._torch_cuda else torch.device("cpu") if _TORCH_AVAILABLE else None
        self._torch_use_fp16 = False
        self._torch_dtype = (torch.float16 if (self._torch_cuda and self._torch_use_fp16) else torch.float32) if _TORCH_AVAILABLE else None
        self._ud_torch_grid = None
        self._ud_torch_grid_wh = None
        
        # Alpha for getOptimalNewCameraMatrix
        self.alpha = 0.0
    
    def load_calibration(self, path):
        """Load camera calibration from npz file"""
        try:
            cal = np.load(str(path), allow_pickle=True)
            self._ud_model = str(cal["model"])
            self._ud_K = cal["K"].astype(np.float32)
            self._ud_D = cal["D"].astype(np.float32)
            self._ud_img_size = tuple(int(x) for x in cal["img_size"])
            self._ud_src_size = None
            self._ud_m1 = self._ud_m2 = None
            self._ud_gm1 = self._ud_gm2 = None
            self._ud_torch_grid = None
            self._ud_torch_grid_wh = None
            print(f"[ImageProcessor] Loaded: model={self._ud_model}, img_size={self._ud_img_size}, cv2.cuda={self._use_cv2_cuda}, torch={self._torch_cuda}")
            return True
        except Exception as e:
            print(f"[ImageProcessor] Load failed: {e}")
            return False
    
    def has_calibration(self):
        """Check if calibration is loaded"""
        return self._ud_K is not None
    
    def _scale_K(self, K, sx, sy):
        """Scale camera matrix K"""
        K2 = K.copy()
        K2[0,0] *= sx
        K2[1,1] *= sy
        K2[0,2] *= sx
        K2[1,2] *= sy
        K2[2,2] = 1.0
        return K2
    
    def _ensure_ud_maps(self, w: int, h: int):
        """Ensure undistortion maps are created for given size"""
        if self._ud_K is None or self._ud_D is None or self._ud_model is None:
            return
        if self._ud_src_size == (w, h) and self._ud_m1 is not None:
            return
        
        Wc, Hc = self._ud_img_size
        sx, sy = w / float(Wc), h / float(Hc)
        K = self._scale_K(self._ud_K, sx, sy)
        D = self._ud_D
        a = float(self.alpha)
        
        if self._ud_model == "pinhole":
            newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=a, newImgSize=(w, h))
            self._ud_m1, self._ud_m2 = cv2.initUndistortRectifyMap(
                K, D, None, newK, (w, h), cv2.CV_32FC1
            )
        elif self._ud_model == "fisheye":
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (w, h), None, balance=a
            )
            self._ud_m1, self._ud_m2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, None, newK, (w, h), cv2.CV_32FC1
            )
        
        self._ud_src_size = (w, h)
        
        # CUDA maps
        if self._use_cv2_cuda and self._ud_m1 is not None:
            try:
                self._ud_gm1 = cv2.cuda_GpuMat()
                self._ud_gm2 = cv2.cuda_GpuMat()
                self._ud_gm1.upload(self._ud_m1)
                self._ud_gm2.upload(self._ud_m2)
            except Exception as e:
                print(f"[ImageProcessor] CUDA upload failed: {e}")
                self._ud_gm1 = self._ud_gm2 = None
    
    def _ensure_torch_grid(self, h: int, w: int):
        """Ensure torch grid for GPU undistortion"""
        if not self._torch_available:
            return
        if self._ud_torch_grid is not None and self._ud_torch_grid_wh == (w, h):
            return
        
        self._ensure_ud_maps(w, h)
        if self._ud_m1 is None:
            return
        
        import torch
        import torch.nn.functional as F
        
        m1_t = torch.from_numpy(self._ud_m1).float()
        m2_t = torch.from_numpy(self._ud_m2).float()
        
        grid_x = (m1_t / (w - 1)) * 2 - 1
        grid_y = (m2_t / (h - 1)) * 2 - 1
        self._ud_torch_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        if self._torch_cuda:
            self._ud_torch_grid = self._ud_torch_grid.to(self._torch_device)
        
        self._ud_torch_grid_wh = (w, h)
    
    def undistort(self, img, use_torch=False):
        """Undistort a BGR image using best available method"""
        if self._ud_K is None or img is None:
            return img
        
        h, w = img.shape[:2]
        
        # Torch acceleration (fastest)
        if use_torch and self._torch_available and self._torch_cuda:
            return self._undistort_torch(img, h, w)
        
        # CUDA acceleration
        if self._use_cv2_cuda:
            return self._undistort_cuda(img, w, h)
        
        # CPU fallback
        self._ensure_ud_maps(w, h)
        if self._ud_m1 is None:
            return img
        return cv2.remap(img, self._ud_m1, self._ud_m2, cv2.INTER_LINEAR)
    
    def _undistort_cuda(self, img, w, h):
        """Undistort using CUDA"""
        self._ensure_ud_maps(w, h)
        if self._ud_gm1 is None:
            return img
        
        try:
            gpu_src = cv2.cuda_GpuMat()
            gpu_src.upload(img)
            gpu_dst = cv2.cuda.remap(gpu_src, self._ud_gm1, self._ud_gm2, cv2.INTER_LINEAR)
            return gpu_dst.download()
        except Exception as e:
            print(f"[ImageProcessor] CUDA remap failed: {e}")
            return cv2.remap(img, self._ud_m1, self._ud_m2, cv2.INTER_LINEAR)
    
    def _undistort_torch(self, img, h, w):
        """Undistort using Torch"""
        import torch
        import torch.nn.functional as F
        
        self._ensure_torch_grid(h, w)
        if self._ud_torch_grid is None:
            return img
        
        try:
            # BGR -> RGB, HWC -> CHW
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(
                dtype=self._torch_dtype, device=self._torch_device
            )
            img_t = img_t / 255.0
            
            out_t = F.grid_sample(img_t, self._ud_torch_grid, mode='bilinear', 
                                  padding_mode='border', align_corners=True)
            
            out_t = (out_t * 255.0).clamp(0, 255).squeeze(0).permute(1, 2, 0)
            out_np = out_t.cpu().to(torch.uint8).numpy()
            out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
            return out_bgr
        except Exception as e:
            print(f"[ImageProcessor] Torch undistort failed: {e}")
            return self.undistort(img, use_torch=False)
    
    def undistort_pair(self, img_on, img_off, use_torch=False):
        """Undistort a pair of images"""
        if self._ud_K is None:
            return img_on, img_off
        img_on = self.undistort(img_on, use_torch=use_torch)
        img_off = self.undistort(img_off, use_torch=use_torch)
        return img_on, img_off
    
    def load_image(self, path):
        """Load image from file"""
        try:
            nparr = np.fromfile(str(path), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"[ImageProcessor] Image load error: {e}")
            return None
    
    def load_image_pair(self, path_on, path_off):
        """Load a pair of images"""
        img_on = self.load_image(path_on)
        img_off = self.load_image(path_off)
        return img_on, img_off
