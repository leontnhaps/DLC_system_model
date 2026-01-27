import cv2
import numpy as np
import pathlib
from tkinter import Tk, filedialog

def load_image(path):
    """한글 경로 지원 이미지 로드"""
    try:
        stream = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(stream, cv2.IMREAD_COLOR)
    except Exception:
        return None

def save_image(path, img):
    """한글 경로 지원 이미지 저장"""
    try:
        ext = path.suffix
        result, encoded_img = cv2.imencode(ext, img)
        if result:
            with open(path, "wb") as f:
                encoded_img.tofile(f)
            return True
    except Exception as e:
        print(f"저장 실패: {e}")
    return False

def visualize_sahi_tiling(img, rows=2, cols=3, overlap=0.25):
    """
    SAHI 타일링 방식을 시각화
    rows, cols: 행/열 개수 (2x3 = 6등분)
    overlap: 타일 간 겹치는 비율 (0.25 = 25%)
    """
    H, W = img.shape[:2]
    
    # 타일 크기 계산 (겹침 포함)
    tile_h = int(H / rows)
    tile_w = int(W / cols)
    
    # 겹침 크기
    ov_h = int(tile_h * overlap)
    ov_w = int(tile_w * overlap)
    
    # 실제 타일 크기 (겹침 포함)
    step_h = tile_h - ov_h
    step_w = tile_w - ov_w
    
    # 시각화용 이미지 복사
    vis_img = img.copy()
    
    # 타일 좌표 생성 및 그리기
    tiles = []
    tile_num = 0
    
    for y in range(0, H, step_h):
        for x in range(0, W, step_w):
            # 타일 영역 계산
            y2 = min(y + tile_h, H)
            x2 = min(x + tile_w, W)
            # 마지막 타일 크기 조정
            y1 = max(0, y2 - tile_h)
            x1 = max(0, x2 - tile_w)
            
            tiles.append((x1, y1, x2, y2))
            
            # 타일 경계 그리기 (반투명 효과)
            overlay = vis_img.copy()
            
            # 타일 영역 색상 채우기 (번갈아가며)
            color = (255, 100, 100) if tile_num % 2 == 0 else (100, 100, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
            # 반투명 효과 적용
            alpha = 0.1
            cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)
            
            # 타일 경계선 그리기
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # 타일 번호 표시
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            text = f"Tile {tile_num}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = cx - text_size[0] // 2
            text_y = cy + text_size[1] // 2
            
            # # 텍스트 배경
            # cv2.rectangle(vis_img, 
            #              (text_x - 10, text_y - text_size[1] - 10),
            #              (text_x + text_size[0] + 10, text_y + 10),
            #              (0, 0, 0), -1)
            
            # 텍스트
            #cv2.putText(vis_img, text, (text_x, text_y), 
            #           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            # 타일 크기 정보
            size_text = f"{x2-x1}x{y2-y1}"
            #cv2.putText(vis_img, size_text, (x1 + 10, y1 + 30), 
            #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            tile_num += 1
            
            if x2 >= W: break
        if y2 >= H: break
    
    # # 이미지 정보 오버레이
    # info_text = [
    #     f"Image Size: {W}x{H}",
    #     f"Grid: {rows}x{cols}",
    #     f"Overlap: {overlap*100:.0f}%",
    #     f"Tile Size: ~{tile_w}x{tile_h}",
    #     f"Total Tiles: {len(tiles)}"
    # ]
    
    # y_offset = 30
    # for i, text in enumerate(info_text):
    #     # 배경
    #     text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    #     cv2.rectangle(vis_img, 
    #                  (10, y_offset - 25 + i*35),
    #                  (text_size[0] + 20, y_offset + 10 + i*35),
    #                  (0, 0, 0), -1)
        
    #     # 텍스트
    #     cv2.putText(vis_img, text, (15, y_offset + i*35), 
    #                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return vis_img, tiles

def main():
    # Tkinter 초기화
    root = Tk()
    root.withdraw()

    print("=== SAHI 타일링 시각화 도구 ===")
    print("이미지가 어떻게 분할되는지 확인합니다.")

    # 이미지 폴더 선택
    print("\n[1] 시각화할 이미지가 있는 폴더를 선택하세요...")
    target_dir = filedialog.askdirectory(title="이미지 폴더 선택")
    if not target_dir:
        print("취소되었습니다.")
        return
    
    target_path = pathlib.Path(target_dir)

    # 결과 저장 폴더
    save_dir = target_path / "sahi_tiling_visualization"
    save_dir.mkdir(exist_ok=True)

    # 이미지 파일 찾기
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(target_path.glob(ext))
        files.extend(target_path.glob(ext.upper()))
    files = sorted(list(set(files)))
    
    if not files:
        print("이미지가 없습니다.")
        return

    print(f"\n[2] 총 {len(files)}장 처리 시작...")
    
    # 타일링 설정
    rows = 2
    cols = 3
    overlap = 0.15

    # 처리
    count = 0
    for i, fpath in enumerate(files):
        try:
            print(f"  [{i+1}/{len(files)}] {fpath.name} 처리 중...", end="", flush=True)
            
            img = load_image(fpath)
            if img is None:
                print(" 건너뜀 (읽기 실패)")
                continue

            # 타일링 시각화
            vis_img, tiles = visualize_sahi_tiling(img, rows, cols, overlap)

            # 저장
            save_path = save_dir / f"tiled_{fpath.name}"
            if not save_image(save_path, vis_img):
                print(" 저장 실패")
                continue
            
            count += 1
            print(f" 완료 ({len(tiles)}개 타일)")

        except Exception as e:
            print(f" 에러: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n=== 완료! ===")
    print(f"처리된 이미지: {count}장")
    print(f"저장 위치: {save_dir}")

if __name__ == "__main__":
    main()
