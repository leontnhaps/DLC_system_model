import os
import cv2
import torch
import pathlib
from ultralytics import YOLO
from tkinter import Tk, filedialog

def draw_results(img, results):
    """YOLO 결과 그리기"""
    plot_img = img.copy()
    
    if results.boxes:
        for box in results.boxes:
            # 박스 좌표
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 신뢰도 및 클래스
            conf = float(box.conf.cpu().numpy().item())
            cls = int(box.cls.cpu().numpy().item())
            
            # 박스 그리기
            cv2.rectangle(plot_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 라벨
            label = f"{conf:.2f}"
            cv2.putText(plot_img, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return plot_img

def run_yolo_on_folder():
    # 1. Tkinter 초기화
    root = Tk()
    root.withdraw()

    print("=== YOLO 표준 분석 도구 ===")
    print("전체 이미지를 한 번에 분석합니다 (타일링 없음)")

    # 2. YOLO 가중치 파일 선택
    # 편의를 위해 경로 고정 (필요시 주석 해제하여 선택창 사용)
    model_path = r"C:\Users\gmlwn\OneDrive\바탕 화면\ICon1학년\OpticalWPT\PTCamera_waveshare\yolov11m_diff.pt"
    if not os.path.exists(model_path):
        print("\n[1] YOLO 가중치 파일(.pt)을 선택하세요...")
        model_path = filedialog.askopenfilename(
            title="YOLO 가중치 파일 선택",
            filetypes=[("YOLO weights", "*.pt"), ("All files", "*.*")]
        )
        if not model_path: return

    print(f"-> 모델: {model_path}")

    # 3. 테스트할 이미지 폴더 선택
    print("\n[2] 테스트할 이미지가 있는 폴더를 선택하세요...")
    target_dir = filedialog.askdirectory(title="이미지 폴더 선택")
    if not target_dir: return
    target_path = pathlib.Path(target_dir)

    # 4. 결과 저장 폴더
    save_dir = target_path / "yolo_standard_results"
    save_dir.mkdir(exist_ok=True)

    # 5. 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[3] 모델 로딩 중... (Device: {device})")
    model = YOLO(model_path)

    # 6. 이미지 파일 찾기
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(target_path.glob(ext))
        files.extend(target_path.glob(ext.upper()))
    files = sorted(list(set(files)))
    
    if not files:
        print("이미지가 없습니다.")
        return

    print(f"\n[4] 총 {len(files)}장 처리 시작 (표준 YOLO 추론)...")

    # 7. 실행
    count = 0
    for i, fpath in enumerate(files):
        try:
            img = cv2.imread(str(fpath))
            if img is None: continue

            # ★ 표준 YOLO 추론 (타일링 없음) ★
            print(f"  -> [{i+1}/{len(files)}] {fpath.name} 분석 중...", end="", flush=True)
            
            results = model.predict(
                img, 
                conf=0.20,      # 신뢰도 임계값
                iou=0.45,       # NMS IoU 임계값
                device=device,
                verbose=False
            )[0]

            # 결과 개수
            num_detections = len(results.boxes) if results.boxes else 0

            # 결과 그리기
            res_img = draw_results(img, results)

            # 저장
            save_path = save_dir / f"res_{fpath.name}"
            cv2.imwrite(str(save_path), res_img)
            
            count += 1
            print(f" 완료 ({num_detections}개 검출)")

        except Exception as e:
            print(f"에러 ({fpath.name}): {e}")
            import traceback
            traceback.print_exc()

    print(f"\n=== 완료! 저장 위치: {save_dir} ===")

if __name__ == "__main__":
    run_yolo_on_folder()
