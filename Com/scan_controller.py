"""
Scan Controller - Manages scan session, image saving, and progress tracking
"""
import pathlib
from datetime import datetime


class ScanController:
    """Scan 세션 관리 및 이미지 저장"""
    
    def __init__(self, save_dir):
        """
        Args:
            save_dir: 베이스 저장 디렉토리 (pathlib.Path)
        """
        self.save_dir = pathlib.Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Scan state
        self.active = False
        self.session = None
        self.session_dir = None
        self.total = 0
        self.done = 0
    
    def start_session(self):
        """새 스캔 세션 시작"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session = f"scan_{ts}"
        self.session_dir = self.save_dir / self.session
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.active = True
        self.total = 0
        self.done = 0
        
        print(f"[SCAN] Session started: {self.session}")
        print(f"[SCAN] Save dir: {self.session_dir}")
        
        return self.session
    
    def stop_session(self):
        """스캔 세션 종료"""
        self.active = False
        print(f"[SCAN] Session stopped: {self.session}")
        
        result = {
            'session': self.session,
            'session_dir': self.session_dir,
            'total': self.total,
            'done': self.done
        }
        return result
    
    def save_image(self, name, data):
        """이미지 저장
        
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
        
        save_path = self.session_dir / name
        save_path.write_bytes(data)
        print(f"[SCAN] Saved: {name}")
        
        return save_path
    
    def update_progress(self, done, total):
        """Progress 업데이트
        
        Args:
            done: 완료된 포인트 수
            total: 전체 포인트 수
        """
        self.done = done
        self.total = total
    
    def get_progress(self):
        """Progress 반환
        
        Returns:
            (done, total) 튜플
        """
        return (self.done, self.total)
    
    def is_active(self):
        """스캔 진행 중인지 확인"""
        return self.active
    
    def get_session_name(self):
        """현재 세션 이름 반환"""
        return self.session
