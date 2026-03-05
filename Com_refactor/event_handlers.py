#!/usr/bin/env python3
"""
Event handlers mixin
Handles all event processing from ui_q
"""

import queue
import time
from tkinter import messagebox
from network import ui_q, pop_latest_preview


class EventHandlersMixin:
    """이벤트 처리 믹스인 - _poll 및 이벤트 핸들링"""
    
    def _poll(self):
        """통합 이벤트 루프 - ui_q에서 모든 이벤트 처리"""
        q = self.bus if hasattr(self, "bus") else ui_q
        try:
            while True:
                tag, payload = q.get_nowait()
                
                if tag == "evt":
                    self._handle_event(payload)
                elif tag == "preview":
                    self._set_preview(payload)
                elif tag == "saved":
                    self._handle_saved_image(payload)
                elif tag == "toast":
                    print(f"[TOAST] {payload}")
        except queue.Empty:
            pass

        # done 이후 tail 이미지 반영이 끝나면 finalize
        if hasattr(self, '_maybe_finalize_scan'):
            self._maybe_finalize_scan()

        # Preview는 최신 1프레임만 표시 (큐 누적 방지)
        preview = pop_latest_preview()
        if preview is not None:
            self._set_preview(preview)
        
        # Poll interval (50ms)
        self.root.after(50, self._poll)
    
    def _handle_event(self, event):
        """서버 이벤트 처리 (start/progress/done/error)"""
        evt = event.get("event")
        
        if evt == "start":
            total = event.get("total", 0)
            self.scan_ctrl.update_progress(0, total)
            self.scan_tab.prog.configure(value=0, maximum=total)
            self.scan_tab.prog_lbl.config(text=f"0 / {total}")
            print(f"[EVENT] Scan started: {total} images")
        
        elif evt == "progress":
            done = event.get("done", 0)
            total = event.get("total", 100)
            name = event.get("name", "")
            
            self.scan_ctrl.update_progress(done, total)
            self.scan_tab.prog.configure(value=done, maximum=total)
            self.scan_tab.prog_lbl.config(text=f"{done} / {total}")
            print(f"[EVENT] Progress: {done}/{total} - {name}")
        
        elif evt == "done":
            done, total = self.scan_ctrl.get_progress()
            print(f"[EVENT] Scan completed: {done}/{total}")
            self.info_label.config(text=f"✅ 스캔 완료: {done}/{total}")
            
            # done 즉시 stop하지 않고 tail 이미지 유입이 멈출 때 finalize
            self._scan_done_pending = True
            if not getattr(self, '_last_scan_image_ts', 0.0):
                self._last_scan_image_ts = time.monotonic()
            print("[EVENT] Scan done -> waiting for tail images before finalizing")
        
        elif evt == "error":
            msg = event.get("message", "Unknown error")
            
            # "no agent connected"는 무시 (Rasp 대기 중)
            if "no agent connected" in msg.lower():
                return
            
            print(f"[EVENT] Error: {msg}")
            self.info_label.config(text=f"❌ 오류: {msg}")
            messagebox.showerror("Scan Error", msg)
    
    def _set_preview(self, img_bytes: bytes):
        """프리뷰 이미지 표시"""
        try:
            self.preview_frame.display_image(img_bytes)
            self.frame_count += 1
        except Exception as e:
            print(f"[PREVIEW] Error: {e}")
    
    def _handle_saved_image(self, payload):
        """저장된 이미지 처리"""
        name, data = payload
        from image_router import route_saved_image

        try:
            route_saved_image(self, name, data)
            return
        except Exception as e:
            print(f"[ROUTER] route_saved_image fallback: {e}")

        # Legacy fallback path (preserve original inline behavior)
        if hasattr(self, '_aiming_active') and self._aiming_active:
            if name.startswith("pointing_"):
                print(f"[POINTING_IMG] Routing to pointing handler: {name}")
                self._on_pointing_image_received(name, data)
                self._set_preview(data)
                return

        if hasattr(self, "_notify_blocking_snap_saved"):
            self._notify_blocking_snap_saved(name, data)

        if self.scan_ctrl.is_active():
            saved_path = self.scan_ctrl.save_image(name, data)
            if saved_path:
                print(f"[SCAN_SAVE] {saved_path}")
                self._last_scan_image_ts = time.monotonic()
        else:
            from pathlib import Path
            SAVE_DIR = Path("captures")
            SAVE_DIR.mkdir(exist_ok=True)
            save_path = SAVE_DIR / name
            with open(save_path, 'wb') as f:
                f.write(data)
            print(f"[SAVE] {save_path}")
            self.info_label.config(text=f"💾 저장됨: {name}")

            if name.startswith("snap_") and hasattr(self, '_on_manual_snap_saved'):
                self._on_manual_snap_saved(name)

        self._set_preview(data)
