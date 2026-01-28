#!/usr/bin/env python3
"""
Event handlers mixin
Handles all event processing from ui_q
"""

import time
import queue
import cv2
import numpy as np
from PIL import Image, ImageTk
from network import ui_q


class EventHandlersMixin:
    """Event processing and polling logic"""
    
    # ========== Poll Loop ==========
    
    def _poll(self):
        """Main event loop - check triggers and process events"""
        self._check_pointing_trigger()
        
        # PV status 체크 제거됨 (아두이노)
        
        try:
            while True:
                tag, payload = ui_q.get_nowait()
                
                if tag == "evt":
                    self._handle_server_event(payload)
                elif tag == "preview":
                    self._set_preview(payload)
                elif tag == "saved":
                    self._handle_saved_image(payload)
                elif tag == "toast":
                    print(f"[TOAST] {payload}")
                elif tag == "pointing_step_2":
                    self._handle_pointing_step2()
                elif tag == "preview_on":
                    self._handle_preview_on()
                elif tag == "debug_preview":
                    self._update_debug_preview(payload)
        except queue.Empty:
            pass
        
        # Poll interval constant (to avoid circular import)
        POLL_INTERVAL_MS = 60
        self.root.after(POLL_INTERVAL_MS, self._poll)
    
    def _check_pointing_trigger(self):
        """Check if pointing cycle should trigger"""
        if not self.pointing_enable.get():
            return
        if self._pointing_state != 0:
            return
        
        now_ms = time.time() * 1000
        if (now_ms - self._pointing_last_ts) < float(self.pointing_interval.get() * 1000):
            return
        
        self._start_pointing_cycle()
    


    def _check_pv_status(self):
        """
        Check laser status to control PV recording and throttle UI updates
        """
        # 1. Check Laser State
        # Manual laser toggle OR Pointing logic where laser is firing
        # Pointing states: 1(None->LaserOn), 2(LaserOn->LaserOff) -> roughly implies laser is active or about to be
        # However, purely checking self.laser_on.get() handles the manual case.
        # For Pointing, the script toggles 'laser' cmd. It might NOT update self.laser_on boolean var if driven by script commands directly?
        # Actually EventHandlers updates self.ctrl.send, but doesn't necessarily update a local BooleanVar for pointing steps.
        # Let's rely on self.laser_on.get() for manual, and infer pointing.
        
        is_manual_laser = self.laser_on.get()
        # State 1 = WAIT_LASER_ON (Laser is ON).
        # State 2 = WAIT_LASER_OFF (Laser is OFF, waiting for captured image).
        # So we should ONLY record during State 1.
        is_pointing_laser = (self._pointing_state == 1)
        
        is_laser_active = is_manual_laser or is_pointing_laser
        self.pv_monitor.set_recording(is_laser_active)
        
        # 2. Throttle UI Updates (Graph)
        if not hasattr(self, '_last_pv_update'):
            self._last_pv_update = 0
            
        now = time.time()
        if (now - self._last_pv_update) > 1.0:  # Update every 500ms
            self.update_pv_graph()
            self._last_pv_update = now
    
    # ========== Server Event Handlers ==========
    
    def _handle_server_event(self, evt: dict):
        """Handle server event messages"""
        kind = evt.get("event","")
        
        if kind == "hello":
            self._handle_hello_event(evt)
        elif kind == "start":
            self._handle_scan_start(evt)
        elif kind == "progress":
            self._handle_scan_progress(evt)
        elif kind == "done":
            self._handle_scan_done(evt)
        else:
            print(f"[EVT] {evt}")
    
    def _handle_hello_event(self, evt):
        """Handle hello event from server - preview auto start"""
        # Preview auto start (simplified - always try if enabled)
        if self.preview_enable.get():
            self.toggle_preview()
    
    def _handle_scan_start(self, evt):
        """Handle scan start event"""
        session = evt.get("session", "")
        total = int(evt.get("total", 0))
        
        # Progress/label updates (from original)
        if hasattr(self, 'prog'):
            self.prog.configure(maximum=max(1, total), value=0)
        self.prog_lbl.config(text=f"0 / {total}")
        if hasattr(self, 'dl_lbl'):
            self.dl_lbl.config(text="DL 0")
        self.last_lbl.config(text="Last: -")
        
        # Start scan controller
        yolo_path = self.yolo_wpath.get().strip()
        if yolo_path and self.scan_controller:
            self.scan_controller.start_scan(session, yolo_path)
    
    def _handle_scan_progress(self, evt):
        """Handle scan progress event"""
        cnt = evt.get("done", 0)
        tot = evt.get("total", 1)
        name = evt.get("name", "")
        
        if hasattr(self, 'prog'):
            self.prog.configure(value=cnt)
        self.prog_lbl.config(text=f"{cnt} / {tot}")
        if name:
            self.last_lbl.config(text=f"Last: {name}")
    
    def _handle_scan_done(self, evt):
        """Handle scan done event"""
        # Stop scan controller
        if self.scan_controller:
            result = self.scan_controller.stop_scan()
            csv_path = result.get('csv_path', '')
            detected = result.get('detected', 0)
            
            print(f"[SCAN] CSV: {csv_path}, Detected: {detected}")
            
            # Auto-load CSV to Pointing tab and compute (원본 로직!)
            if csv_path and csv_path != 'unknown':
                self.point_csv_path.set(str(csv_path))
                print(f"[DEBUG scan_done] CSV auto-loaded to Pointing tab: {csv_path}")
                self.pointing_compute()
                ui_q.put(("toast", f"📄 CSV 자동 로드됨: {csv_path}"))
        
        # Resume preview
        ui_q.put(("preview_on", None))
    
    # ========== Image Handlers ==========
    
    def _handle_saved_image(self, payload):
        """Handle saved image from server"""
        name, data = payload
        
        # Delegate to scan controller if active
        if self.scan_controller:
            data = self.scan_controller.on_image_received(name, data)
        
        # Pointing logic
        if name == "pointing_laser_on.jpg":
            self._handle_pointing_laser_on(data)
        elif name == "pointing_laser_off.jpg":
            self._handle_pointing_laser_off(data)
        elif name == "pointing_led_on.jpg":
            self._handle_pointing_led_on(data)
        elif name == "pointing_led_off.jpg":
            self._handle_pointing_led_off(data)
        else:
            self._handle_generic_saved_image(name, data)
    
    def _handle_pointing_laser_on(self, data):
        """Handle laser ON image for pointing"""
        if self._pointing_state != 1:
            return
        self._pointing_state = 2
        self._pointing_img_laser_on = data
        self._set_preview(data)  # Show Laser ON image
        
        # Trigger laser OFF
        self.ctrl.send({"cmd": "laser", "value": 0})
        wait_ms = int(self.point_settle.get() * 1000)
        self.root.after(wait_ms, lambda: self.ctrl.send({
            "cmd": "snap", "width": self.width.get(), "height": self.height.get(),
            "quality": self.quality.get(), "save": "pointing_laser_off.jpg",
            "ud_save": self.ud_save_copy.get()
        }))
    
    def _handle_pointing_laser_off(self, data):
        """Handle laser OFF image for pointing"""
        if self._pointing_state != 2:
            return
        self._pointing_state = 3
        self._pointing_img_laser_off = data
        self._set_preview(data)  # Show Laser OFF image
        
        # Process laser images
        img_on = cv2.imdecode(np.frombuffer(self._pointing_img_laser_on, np.uint8), cv2.IMREAD_COLOR)
        img_off = cv2.imdecode(np.frombuffer(self._pointing_img_laser_off, np.uint8), cv2.IMREAD_COLOR)
        self._run_pointing_laser_logic(img_on, img_off)
    
    def _handle_pointing_step2(self):   
        """Trigger LED ON for object detection"""
        if self._pointing_state != 3:
            return
        self._pointing_state = 4
        
        self.ctrl.send({"cmd": "led", "value": 255})
        wait_ms = int(self.point_settle.get() * 1000)
        self.root.after(wait_ms, lambda: self.ctrl.send({
            "cmd": "snap", "width": self.width.get(), "height": self.height.get(),
            "quality": self.quality.get(), "save": "pointing_led_on.jpg",
            "ud_save": self.ud_save_copy.get()
        }))
    
    def _handle_pointing_led_on(self, data):
        """Handle LED ON image for pointing"""
        if self._pointing_state != 4:
            return
        self._pointing_state = 5
        self._pointing_img_led_on = data
        self._set_preview(data)  # Show LED ON image
        
        self.ctrl.send({"cmd": "led", "value": 0})
        wait_ms = int(self.point_settle.get() * 1000)
        self.root.after(wait_ms, lambda: self.ctrl.send({
            "cmd": "snap", "width": self.width.get(), "height": self.height.get(),
            "quality": self.quality.get(), "save": "pointing_led_off.jpg",
            "ud_save": self.ud_save_copy.get()
        }))
    
    def _handle_pointing_led_off(self, data):
        """Handle LED OFF image for pointing"""
        if self._pointing_state != 5:
            return
        self._pointing_state = 6
        self._pointing_img_led_off = data
        self._set_preview(data)  # Show LED OFF image
        
        # Process object images
        img_on = cv2.imdecode(np.frombuffer(self._pointing_img_led_on, np.uint8), cv2.IMREAD_COLOR)
        img_off = cv2.imdecode(np.frombuffer(self._pointing_img_led_off, np.uint8), cv2.IMREAD_COLOR)
        self._run_pointing_object_logic(img_on, img_off)
    
    def _handle_preview_on(self):
        """Resume preview"""
        self.resume_preview()
    
    def _handle_generic_saved_image(self, name, data):
        """Handle generic saved image - update preview and optionally save undistorted copy"""
        # Update preview with received image
        self._set_preview(data)
        
        # [NEW] Save undistorted copy if enabled
        if self.ud_save_copy.get() and self.image_processor.has_calibration():
            try:
                # Decode image
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    # Undistort
                    img_ud = self._undistort_bgr(img)
                    
                    # Generate .ud filename
                    import pathlib
                    from datetime import datetime
                    outdir = pathlib.Path(self.outdir.get())
                    outdir.mkdir(parents=True, exist_ok=True)
                    
                    # Replace extension with .ud.jpg
                    base_name = pathlib.Path(name).stem
                    ud_name = f"{base_name}.ud.jpg"
                    ud_path = outdir / ud_name
                    
                    # Save undistorted image
                    cv2.imwrite(str(ud_path), img_ud, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    print(f"[SAVED UD] {ud_name}")
            except Exception as e:
                print(f"[ERROR] Failed to save undistorted copy: {e}")
        
        print(f"[SAVED] {name}")
        
        # [NEW] Resume preview after snap if it was enabled
        if self._resume_preview_after_snap:
            self._resume_preview_after_snap = False
            self.root.after(200, self.resume_preview)
    
    # ========== Preview Handlers ==========
    
    def _set_preview(self, img_bytes: bytes):
        """Set preview image"""
        try:
            arr = np.frombuffer(img_bytes, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None: 
                return

            if self.ud_enable.get() and self.image_processor.has_calibration():
                bgr = self._undistort_bgr(bgr)

            # Convert and display
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            self._draw_preview_to_label(im)

        except Exception as e:
            print("[preview] err:", e)
    
    def _draw_preview_to_label(self, im: Image.Image):
        """Draw image to preview label"""
        lbl_w = self.preview_label.winfo_width()
        lbl_h = self.preview_label.winfo_height()
        
        if lbl_w <= 1 or lbl_h <= 1:
            lbl_w, lbl_h = 800, 600
        
        im_w, im_h = im.size
        scale = min(lbl_w/im_w, lbl_h/im_h)
        
        new_w = int(im_w * scale)
        new_h = int(im_h * scale)
        im_resized = im.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        photo = ImageTk.PhotoImage(im_resized)
        self.preview_label.config(image=photo)
        self.preview_label.image = photo
    
    def _update_debug_preview(self, img_array):
        """Update debug preview in Pointing tab"""
        try:
            from PIL import Image
            import cv2
            # img_array is already cropped 400x400 BGR
            rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            
            # Resize to fit 400x400 frame
            im_resized = im.resize((400, 400), Image.Resampling.BILINEAR)
            
            photo = ImageTk.PhotoImage(im_resized)
            self.debug_preview_label.config(image=photo)
            self.debug_preview_label.image = photo
        except Exception as e:
            print(f"[DEBUG preview] Error: {e}")
