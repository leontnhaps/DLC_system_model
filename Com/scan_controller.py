"""
Scan Controller - Manages scan session, YOLO processing, CSV export, and MOT
Worker Thread pattern for async YOLO processing + Multi-Object Tracking
"""
import pathlib
import csv
import re
import threading
import queue
import os
from datetime import datetime
import cv2
import numpy as np
from mot import ObjectTracker


class ScanController:
    """실시간 스캔 처리 관리자 - Worker Thread pattern"""
    def __init__(self, save_dir, yolo_processor=None):
        """
        Args:
            save_dir: 베이스 저장 디렉토리 (pathlib.Path)
            yolo_processor: YOLOProcessor 인스턴스 (optional)
        """
        self.save_dir = pathlib.Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.yolo_processor = yolo_processor
        
        # MOT Tracker
        self.mot_tracker = ObjectTracker(roi_size=300, grid_size=(11, 11))
        
        # Scan state
        self.active = False
        self.session = None
        self.session_dir = None
        self.total = 0
        self.done = 0
        
        # YOLO 설정
        self.yolo_weights_path = None
        
        # Real-time processing buffer: (pan, tilt) -> {'on': img, 'off': img}
        self.image_pairs = {}
        
        # CSV writer
        self.csv_writer = None
        self.csv_file = None
        self.csv_path = None
        
        # Statistics
        self.processed_count = 0
        self.detected_count = 0
        self.track_count = 0
        
        # Worker Thread
        self.processing_queue = queue.Queue(maxsize=50)
        self.worker_thread = None
        self.worker_running = False
    
    def _start_worker_thread(self):
        """Worker Thread 시작"""
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.worker_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("[ScanController] Worker thread started")
    
    def _stop_worker_thread(self):
        """Worker Thread 중지"""
        self.worker_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        print("[ScanController] Worker thread stopped")
    
    def _worker_loop(self):
        """Worker thread loop - 이미지 쌍을 비동기로 처리"""
        while self.worker_running:
            try:
                task = self.processing_queue.get(timeout=0.5)
                if task is None:
                    break
                pan, tilt, pair = task
                self._process_pair(pan, tilt, pair)
                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ScanController] Worker error: {e}")
    
    def _process_pair(self, pan, tilt, pair):
        """Worker Thread 내부에서 실행됨 - YOLO 처리"""
        from yolo_utils import predict_with_tiling
        
        # YOLO constants
        YOLO_TILE_ROWS = 2
        YOLO_TILE_COLS = 3
        YOLO_TILE_OVERLAP = 0.15
        YOLO_CONF_THRESHOLD = 0.20
        YOLO_IOU_THRESHOLD = 0.45
        
        try:
            # Diff 계산 (undistort 없음)
            diff = cv2.absdiff(pair['on'], pair['off'])
            H, W = diff.shape[:2]
            
            # YOLO detection
            if self.yolo_processor and self.yolo_weights_path:
                model = self.yolo_processor.get_model(self.yolo_weights_path)
                if model is not None:
                    device = self.yolo_processor.get_device()
                    
                    boxes, scores, classes = predict_with_tiling(
                        model, diff,
                        rows=YOLO_TILE_ROWS, cols=YOLO_TILE_COLS,
                        overlap=YOLO_TILE_OVERLAP,
                        conf=YOLO_CONF_THRESHOLD, iou=YOLO_IOU_THRESHOLD,
                        device=device
                    )
                    
                    # ⭐ MOT 전에 confidence >= 0.5 필터링 (hungarian_final 기준)
                    MOT_CONF_THRESHOLD = 0.5
                    if boxes:
                        filtered_indices = [i for i, score in enumerate(scores) if score >= MOT_CONF_THRESHOLD]
                        if filtered_indices:
                            boxes = [boxes[i] for i in filtered_indices]
                            scores = [scores[i] for i in filtered_indices]
                            classes = [classes[i] for i in filtered_indices]
                        else:
                            boxes = []
                    
                    # MOT tracking
                    if boxes:
                        # Timestamp 생성 (pan, tilt 기반)
                        timestamp = f"{pan:+04d}_{tilt:+03d}"
                        
                        # MOT에 검출 결과 전달 (LED OFF for feature extraction)
                        track_ids = self.mot_tracker.add_detections(
                            boxes, scores,
                            img_on=pair['off'],  # ⭐ LED OFF!
                            diff=diff,
                            pan=pan, tilt=tilt, timestamp=timestamp
                        )
                        
                        # CSV 저장 (track_id 포함)
                        if self.csv_writer:
                            for i, (x, y, w, h) in enumerate(boxes):
                                self.csv_writer.writerow([
                                    pan, tilt, x+w/2, y+h/2, w, h,
                                    float(scores[i]), int(classes[i]), W, H,
                                    track_ids[i]  # ⭐ track_id 추가
                                ])
                                self.detected_count += 1
                            self.csv_file.flush()
            
            self.processed_count += 1
            
        except Exception as e:
            print(f"[ScanController] Pair processing failed ({pan}, {tilt}): {e}")
    
    def start_session(self, yolo_weights_path=None):
        """새 스캔 세션 시작"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session = f"scan_{ts}"
        self.session_dir = self.save_dir / self.session
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.active = True
        self.total = 0
        self.done = 0
        self.yolo_weights_path = yolo_weights_path
        
        # 통계 초기화
        self.image_pairs.clear()
        self.processed_count = 0
        self.detected_count = 0
        
        # Worker Thread 시작
        self._start_worker_thread()
        
        # MOT 리셋
        self.mot_tracker.reset()
        
        # CSV 생성 (track_id 컬럼 추가)
        if self.yolo_weights_path:
            self.csv_path = self.session_dir / f"{self.session}_detections.csv"
            try:
                self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
                self.csv_writer = csv.writer(self.csv_file)
                self.csv_writer.writerow(["pan_deg", "tilt_deg", "cx", "cy", "w", "h", "conf", "cls", "W", "H", "track_id"])
                print(f"[ScanController] CSV created: {self.csv_path}")
            except Exception as e:
                print(f"[ScanController] CSV creation failed: {e}")
        
        print(f"[SCAN] Session started: {self.session}")
        print(f"[SCAN] Save dir: {self.session_dir}")
        if yolo_weights_path:
            print(f"[SCAN] YOLO weights: {yolo_weights_path}")
        
        return self.session
    
    def stop_session(self):
        """스캔 세션 종료 + MOT 후처리"""
        self.active = False
        
        # Worker Thread 중지
        self._stop_worker_thread()
        
        # MOT 후처리 (track 병합)
        if self.yolo_weights_path:
            print("\n" + "="*60)
            print("⭐ MOT 후처리 시작")
            print("="*60)
            merge_map = self.mot_tracker.merge_similar_tracks(
                merge_threshold=0.4,
                min_detections=3
            )
            self.track_count = self.mot_tracker.get_track_count()
            print(f"[MOT] 최종 Track 수: {self.track_count}개")
            print(f"[MOT] 병합된 track: {len(merge_map)}개")
            
            # ⭐ 유사도 로그 저장 (mot_scan_test_hungarian_final.py와 동일 포맷)
            if self.session_dir:
                log_path = self.session_dir / "similarity_log_live.txt"
                print(f"[MOT] Saving similarity log to {log_path}")
                self.mot_tracker.save_similarity_log(str(log_path))
        
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            print(f"[SCAN] Saved to {self.csv_path}")
            csv_abs_path = os.path.abspath(self.csv_path)
        else:
            csv_abs_path = None
        
        self.image_pairs.clear()
        
        print(f"[SCAN] Session stopped: {self.session}")
        print(f"[SCAN] Processed: {self.processed_count}, Detected: {self.detected_count}, Tracks: {self.track_count}")
        
        result = {
            'session': self.session,
            'session_dir': self.session_dir,
            'total': self.total,
            'done': self.done,
            'processed': self.processed_count,
            'detected': self.detected_count,
            'tracks': self.track_count,
            'csv_path': self.csv_path,
            'csv_path_abs': csv_abs_path  # ⭐ 절대 경로 추가
        }
        return result
    
    def save_image(self, name, data):
        """이미지 저장 및 YOLO 처리 큐에 추가
        
        Args:
            name: 파일명
            data: JPEG 바이너리 데이터
        
        Returns:
            저장된 경로 (pathlib.Path) or None
        """
        if not self.active or not self.session_dir:
            return None
        
        # Session 이름이 파일명에 포함되어 있는지 확인
        if self.session not in name:
            return None
        
        # 파일 저장
        save_path = self.session_dir / name
        save_path.write_bytes(data)
        
        # Pan/Tilt 파싱
        match = re.search(r't([+-]?\d+)_p([+-]?\d+)', name)
        if not match:
            return save_path
        
        tilt = int(match.group(1))
        pan = int(match.group(2))
        
        # 디코드
        try:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return save_path
        except Exception as e:
            print(f"[ScanController] Image decode failed: {e}")
            return save_path
        
        # LED ON/OFF 쌍 매칭
        key = (pan, tilt)
        if 'led_on' in name:
            self.image_pairs.setdefault(key, {})['on'] = img
        elif 'led_off' in name:
            self.image_pairs.setdefault(key, {})['off'] = img
        
        # 쌍 완성 체크
        pair = self.image_pairs.get(key, {})
        if 'on' in pair and 'off' in pair:
            # Worker Queue에 추가
            try:
                self.processing_queue.put_nowait((pan, tilt, pair))
                del self.image_pairs[key]
            except queue.Full:
                print(f"[ScanController] Queue full, skipping ({pan}, {tilt})")
                del self.image_pairs[key]
        
        return save_path
    
    def update_progress(self, done, total):
        """Progress 업데이트"""
        self.done = done
        self.total = total
    
    def get_progress(self):
        """Progress 반환"""
        return (self.done, self.total)
    
    def is_active(self):
        """스캔 진행 중인지 확인"""
        return self.active
    
    def get_session_name(self):
        """현재 세션 이름 반환"""
        return self.session
