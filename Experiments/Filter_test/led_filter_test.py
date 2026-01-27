import cv2
import numpy as np
from tkinter import Tk, filedialog

class LEDFilterTest:
    def __init__(self):
        self.image = None
        self.original_image = None  # 원본 이미지 보관
        self.scale_factor = 1.0  # 스케일 팩터
        self.roi_center = None
        self.roi_size = 100
        
        # 화면 크기 설정 (일반적인 모니터 해상도 고려)
        self.max_display_width = 1800
        self.max_display_height = 900
        
        # 파란색 LED 필터 파라미터 (초기값)
        self.h_min = 100
        self.h_max = 130
        self.s_min = 100
        self.s_max = 255
        self.v_min = 150
        self.v_max = 255
        
        # Morphology 파라미터
        self.morph_kernel_size = 5
        
        # 디스플레이 윈도우 이름
        self.window_name = "LED Filter Test"
        self.control_window = "Filter Controls"
        
    def load_image(self, image_path):
        """한글 경로 지원 이미지 로드 및 화면 크기에 맞게 리사이즈"""
        try:
            with open(image_path, 'rb') as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                print("❌ 이미지를 불러올 수 없습니다.")
                return False
            
            # 원본 이미지 저장
            self.original_image = img.copy()
            
            # 이미지 크기 확인
            h, w = img.shape[:2]
            print(f"📐 원본 이미지 크기: {w} x {h}")
            
            # 화면에 맞게 리사이즈 (필요한 경우)
            scale_w = self.max_display_width / w
            scale_h = self.max_display_height / h
            scale = min(scale_w, scale_h, 1.0)  # 1.0보다 크게 확대하지 않음
            
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                self.image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                self.scale_factor = scale
                print(f"🔽 화면에 맞게 리사이즈: {new_w} x {new_h} (스케일: {scale:.2f})")
            else:
                self.image = img
                self.scale_factor = 1.0
                print(f"✅ 원본 크기 그대로 사용")
            
            print(f"✅ 이미지 로드 완료: {self.image.shape}")
            return True
        except Exception as e:
            print(f"❌ 이미지 로드 오류: {e}")
            return False
    
    def mouse_callback(self, event, x, y, flags, param):
        """마우스 클릭 이벤트 핸들러"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_center = (x, y)
            print(f"📍 ROI 중심 선택: ({x}, {y})")
            self.process_and_display()
    
    def get_roi(self):
        """선택된 중심점 기준으로 100x100 ROI 추출"""
        if self.image is None or self.roi_center is None:
            return None
        
        h, w = self.image.shape[:2]
        cx, cy = self.roi_center
        
        # ROI 경계 계산 (이미지 범위 내로 제한)
        half_size = self.roi_size // 2
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(w, cx + half_size)
        y2 = min(h, cy + half_size)
        
        roi = self.image[y1:y2, x1:x2]
        
        return roi, (x1, y1, x2, y2)
    
    def apply_blue_filter(self, roi):
        """파란색 LED 검출 필터 적용"""
        # BGR을 HSV로 변환
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 파란색 범위 마스크 생성
        lower_blue = np.array([self.h_min, self.s_min, self.v_min])
        upper_blue = np.array([self.h_max, self.s_max, self.v_max])
        
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Morphology 연산 (노이즈 제거 및 영역 정리)
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 결과 이미지 (원본에 마스크 적용)
        result = cv2.bitwise_and(roi, roi, mask=mask)
        
        return mask, result
    
    def detect_led_center(self, mask):
        """LED 중심 좌표 검출 (Moments 사용)"""
        M = cv2.moments(mask)
        
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        else:
            return None
    
    def process_and_display(self):
        """ROI 처리 및 결과 시각화"""
        if self.image is None or self.roi_center is None:
            return
        
        # ROI 추출
        roi_data = self.get_roi()
        if roi_data is None:
            return
        
        roi, (x1, y1, x2, y2) = roi_data
        
        # 파란색 필터 적용
        mask, result = self.apply_blue_filter(roi)
        
        # LED 중심 검출
        led_center = self.detect_led_center(mask)
        
        # 시각화용 복사본 생성
        display_img = self.image.copy()
        roi_vis = roi.copy()
        
        # ROI 영역 표시 (전체 이미지)
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # LED 중심 표시 (ROI 내)
        if led_center is not None:
            cx, cy = led_center
            cv2.drawMarker(roi_vis, (cx, cy), (0, 0, 255), 
                          markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
            cv2.putText(roi_vis, f"LED: ({cx},{cy})", (cx + 5, cy - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # 전체 이미지에도 표시
            global_cx = x1 + cx
            global_cy = y1 + cy
            cv2.drawMarker(display_img, (global_cx, global_cy), (0, 0, 255), 
                          markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.putText(display_img, f"Global: ({global_cx},{global_cy})", 
                       (global_cx + 10, global_cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            print(f"✅ LED 검출: ROI({cx}, {cy}) / Global({global_cx}, {global_cy})")
        else:
            cv2.putText(roi_vis, "LED Not Found", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            print("⚠️ LED를 찾지 못했습니다.")
        
        # 결과 이미지 구성
        # ROI 결과들을 수평 배치
        roi_h, roi_w = roi.shape[:2]
        
        # 크기 맞춤을 위해 mask를 3채널로 변환
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # ROI 결과들 수평 배치
        roi_combined = np.hstack([roi_vis, mask_3ch, result])
        
        # ROI 결과에 제목 추가
        roi_with_title = np.zeros((roi_h + 30, roi_w * 3, 3), dtype=np.uint8)
        roi_with_title[30:, :] = roi_combined
        cv2.putText(roi_with_title, "Original", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(roi_with_title, "Mask", (roi_w + 10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(roi_with_title, "Result", (roi_w * 2 + 10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 전체 이미지와 ROI 결과 표시
        cv2.imshow(self.window_name, display_img)
        cv2.imshow("ROI Detailed View", roi_with_title)
    
    def create_controls(self):
        """트랙바를 사용한 파라미터 조절 UI 생성"""
        cv2.namedWindow(self.control_window)
        
        # HSV 파라미터 트랙바
        cv2.createTrackbar("H Min", self.control_window, self.h_min, 179, self.on_trackbar)
        cv2.createTrackbar("H Max", self.control_window, self.h_max, 179, self.on_trackbar)
        cv2.createTrackbar("S Min", self.control_window, self.s_min, 255, self.on_trackbar)
        cv2.createTrackbar("S Max", self.control_window, self.s_max, 255, self.on_trackbar)
        cv2.createTrackbar("V Min", self.control_window, self.v_min, 255, self.on_trackbar)
        cv2.createTrackbar("V Max", self.control_window, self.v_max, 255, self.on_trackbar)
        
        # Morphology 커널 크기 트랙바
        cv2.createTrackbar("Morph Kernel", self.control_window, self.morph_kernel_size, 15, self.on_trackbar)
    
    def on_trackbar(self, val):
        """트랙바 변경 이벤트 핸들러"""
        # 현재 트랙바 값 읽기
        self.h_min = cv2.getTrackbarPos("H Min", self.control_window)
        self.h_max = cv2.getTrackbarPos("H Max", self.control_window)
        self.s_min = cv2.getTrackbarPos("S Min", self.control_window)
        self.s_max = cv2.getTrackbarPos("S Max", self.control_window)
        self.v_min = cv2.getTrackbarPos("V Min", self.control_window)
        self.v_max = cv2.getTrackbarPos("V Max", self.control_window)
        self.morph_kernel_size = max(1, cv2.getTrackbarPos("Morph Kernel", self.control_window))
        
        # Morph kernel은 홀수여야 함
        if self.morph_kernel_size % 2 == 0:
            self.morph_kernel_size += 1
        
        # 결과 업데이트
        self.process_and_display()
    
    def run(self, image_path):
        """메인 실행 함수"""
        # 이미지 로드
        if not self.load_image(image_path):
            return
        
        # 윈도우 생성 및 마우스 콜백 설정
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # 컨트롤 UI 생성
        self.create_controls()
        
        # 초기 이미지 표시
        cv2.imshow(self.window_name, self.image)
        
        print("=" * 60)
        print("📘 LED 필터 테스트 사용법")
        print("=" * 60)
        print("1. 전체 이미지에서 LED가 있는 부분을 마우스 클릭")
        print("2. 100x100 ROI가 자동으로 설정됩니다")
        print("3. 'Filter Controls' 창에서 HSV 파라미터 조절")
        print("4. 'q' 키를 누르면 종료")
        print("=" * 60)
        print(f"\n현재 필터 설정 (파란색 LED):")
        print(f"  H: {self.h_min}-{self.h_max}")
        print(f"  S: {self.s_min}-{self.s_max}")
        print(f"  V: {self.v_min}-{self.v_max}")
        print(f"  Morph Kernel: {self.morph_kernel_size}")
        print("=" * 60)
        
        # 메인 루프
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 프로그램을 종료합니다.")
                break
            elif key == ord('r'):
                # 리셋
                self.roi_center = None
                cv2.imshow(self.window_name, self.image)
                cv2.destroyWindow("ROI Detailed View")
                print("🔄 ROI 선택이 초기화되었습니다.")
        
        cv2.destroyAllWindows()


# --- 실행 예시 ---
if __name__ == "__main__":
    # Tkinter 루트 윈도우 생성 (숨김)
    root = Tk()
    root.withdraw()
    
    # 파일 선택 다이얼로그
    print("📂 이미지 파일을 선택하세요...")
    image_path = filedialog.askopenfilename(
        title="LED 테스트용 이미지 선택",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.bmp"),
            ("All Files", "*.*")
        ]
    )
    
    # 파일 선택 취소 시
    if not image_path:
        print("❌ 이미지 파일이 선택되지 않았습니다.")
    else:
        # LED 필터 테스트 실행
        tester = LEDFilterTest()
        tester.run(image_path)

