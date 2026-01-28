#!/usr/bin/env python3
"""
Scan processing controller
Real-time YOLO detection and CSV logging during scans
"""

import cv2
import numpy as np
import csv
import re
import pathlib
import threading
import queue

# MOT Tracker
from MOT import ObjectTracker

class ScanController:
    """실시간 스캔 처리 관리자 - Worker Thread pattern for async processing"""
    
    def __init__(self, image_processor, yolo_processor, output_dir, mot_roi_size=300, mot_grid_size=(11, 11)):
        """
        Args:
            image_processor: 이미지 처리 객체
            yolo_processor: YOLO 모델 프로세서
            output_dir: 출력 디렉토리
            mot_roi_size: MOT ROI 크기 (픽셀, 중심 기준)
            mot_grid_size: MOT 특징 추출 격자 크기 (rows, cols)
        """
        self.image_processor = image_processor
        self.yolo_processor = yolo_processor
        self.output_dir = output_dir
        
        # Scan state
        self.is_scanning = False
        self.yolo_weights_path = None
        
        # ⭐ MOT Tracker (설정 가능한 파라미터)
        self.mot_tracker = ObjectTracker(roi_size=mot_roi_size, grid_size=mot_grid_size)
        
        # Real-time processing buffer: (pan, tilt) -> {'on': img, 'off': img}
        self.image_pairs = {}
        
        # CSV writer
        self.csv_writer = None
        self.csv_file = None
        self.csv_path = None
        
        # Statistics
        self.processed_count = 0
        self.detected_count = 0
        
        # ===== Worker Thread (핵심 최적화!) =====
        self.processing_queue = queue.Queue(maxsize=50)  # 이미지 처리 큐
        self.worker_thread = None
        self.worker_running = False
    
    def _start_worker_thread(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.worker_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("[ScanController] Worker thread started")
    
    def _stop_worker_thread(self):
        self.worker_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        print("[ScanController] Worker thread stopped")
    
    def _worker_loop(self):
        """Worker thread loop - processes images asynchronously"""
        while self.worker_running:
            try:
                task = self.processing_queue.get(timeout=0.5)
                if task is None: break
                pan, tilt, pair = task
                self._process_pair(pan, tilt, pair)
                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ScanController] Worker error: {e}")
    
    def start_scan(self, session_name, yolo_path):
        self.is_scanning = True
        self.yolo_weights_path = yolo_path
        self.image_pairs.clear()
        self.processed_count = 0
        self.detected_count = 0
        self._start_worker_thread()
        
        # ⭐ MOT Tracker 초기화
        self.mot_tracker.reset()
        
        self.csv_path = self.output_dir / f"{session_name}_detections.csv"
        try:
            self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            # ⭐ track_id 컬럼 추가
            self.csv_writer.writerow(["pan_deg", "tilt_deg", "cx", "cy", "w", "h", "conf", "cls", "track_id", "W", "H"])
            print(f"[ScanController] CSV created: {self.csv_path}")
            return True
        except Exception as e:
            print(f"[ScanController] CSV creation failed: {e}")
            return False
    
    def on_image_received(self, name, data):
        # Save to file
        file_path = self.output_dir / name
        try:
            with open(file_path, 'wb') as f: f.write(data)
        except Exception as e:
            print(f"[ScanController] File save failed: {e}")
        
        # Parse pan/tilt
        match = re.search(r't([+-]?\d+)_p([+-]?\d+)', name)
        if not match: return data
        
        tilt = int(match.group(1))
        pan = int(match.group(2))
        
        if self.is_scanning:
            self._process_realtime(name, data, pan, tilt)
        
        return data
    
    def _process_realtime(self, name, data, pan, tilt):
        # Decode
        try:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None: return
        except Exception as e:
            print(f"[ScanController] Image decode failed: {e}")
            return
        
        # [수정됨] 여기서 undistort 하지 않음! (렉 방지) -> 원본 저장
        key = (pan, tilt)
        if 'led_on' in name:
            self.image_pairs.setdefault(key, {})['on'] = img
        elif 'led_off' in name:
            self.image_pairs.setdefault(key, {})['off'] = img
        
        # Pair complete check
        pair = self.image_pairs.get(key, {})
        if 'on' in pair and 'off' in pair:
            try:
                self.processing_queue.put_nowait((pan, tilt, pair))
                del self.image_pairs[key]
            except queue.Full:
                print(f"[ScanController] Queue full, skipping ({pan}, {tilt})")
                del self.image_pairs[key]
    
    def _process_pair(self, pan, tilt, pair):
        """Worker Thread 내부에서 실행됨"""
        from yolo_utils import predict_with_tiling
        from datetime import datetime
        
        # ⭐ 타임스탬프 생성 (MOT용)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # YOLO constants
        YOLO_TILE_ROWS = 2
        YOLO_TILE_COLS = 3
        YOLO_TILE_OVERLAP = 0.15
        YOLO_CONF_THRESHOLD = 0.20
        YOLO_IOU_THRESHOLD = 0.45
        
        try:
            # [수정됨] 여기서 Undistort 수행 (백그라운드 처리)
            img_on_ud = self.image_processor.undistort(pair['on'], use_torch=True)
            img_off_ud = self.image_processor.undistort(pair['off'], use_torch=True)
            
            # Calculate difference using undistorted images
            diff = cv2.absdiff(img_on_ud, img_off_ud)
            H, W = diff.shape[:2]
            
            # YOLO detection
            model = self.yolo_processor.get_model(self.yolo_weights_path)
            if model is None: return
            
            device = self.yolo_processor.get_device()
            
            boxes, scores, classes = predict_with_tiling(
                model, diff,
                rows=YOLO_TILE_ROWS, cols=YOLO_TILE_COLS,
                overlap=YOLO_TILE_OVERLAP,
                conf=YOLO_CONF_THRESHOLD, iou=YOLO_IOU_THRESHOLD,
                device=device
            )
            
            # ⭐ MOT 추적 - track_id 부여
            if boxes:
                track_ids = self.mot_tracker.add_detections(
                    boxes, scores, img_on_ud, diff, pan, tilt, timestamp
                )
            else:
                track_ids = []
            
            # CSV 저장 (track_id 포함)
            if boxes and self.csv_writer:
                for i, (x, y, w, h) in enumerate(boxes):
                    self.csv_writer.writerow([
                        pan, tilt, x+w/2, y+h/2, w, h,
                        float(scores[i]), int(classes[i]), track_ids[i], W, H
                    ])
                    self.detected_count += 1
                self.csv_file.flush()
            
            self.processed_count += 1
            
        except Exception as e:
            print(f"[ScanController] Pair processing failed ({pan}, {tilt}): {e}")
    
    def stop_scan(self):
        self.is_scanning = False
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        
        self.image_pairs.clear()
        
        return {
            'csv_path': self.csv_path,
            'processed': self.processed_count,
            'detected': self.detected_count
        }