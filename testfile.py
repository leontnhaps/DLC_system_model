import pandas as pd
import matplotlib.pyplot as plt

# 1. 불러올 파일 이름 리스트
file_names = [
    "battery_log_1.csv", 
    "battery_log_2.csv", 
    "battery_log_3.csv", 
    "battery_log_4.csv"
]

plt.figure(figsize=(10, 6))

# 2. 각 파일을 순회하며 그래프 그리기
for i, file in enumerate(file_names):
    try:
        # CSV 파일 읽기 (헤더가 없다고 가정)
        df = pd.read_csv(file, header=None)
        
        # 아두이노에서 출력하는 형식에 따라 처리 (콤마로 시간,전압을 구분하는지)
        if df.shape[1] >= 2:
            # 칼럼이 2개 이상인 경우: 첫 번째 칼럼은 시간(EEPROM 인덱스), 두 번째는 전압
            # errors='coerce'를 사용해 "END" 같은 문자열을 NaN(결측치)으로 변환
            x_data = pd.to_numeric(df[0], errors='coerce')
            y_data = pd.to_numeric(df[1], errors='coerce')
            
            # NaN 값이 들어간 행(예: END가 적힌 줄) 제거
            valid_mask = ~x_data.isna() & ~y_data.isna()
            x_data = x_data[valid_mask]
            y_data = y_data[valid_mask]
            
            plt.plot(x_data, y_data, label=f'Arduino {i+1}', linewidth=1.5)
            
        elif df.shape[1] == 1:
            # 칼럼이 1개인 경우: 값(전압)만 연속으로 들어오는 상황
            y_data = pd.to_numeric(df[0], errors='coerce').dropna()
            # X축은 데이터의 순서(인덱스)로 자동 지정
            plt.plot(y_data.index, y_data, label=f'Arduino {i+1}', linewidth=1.5)

    except FileNotFoundError:
        print(f"⚠️ {file} 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    except Exception as e:
        print(f"⚠️ {file} 처리 중 오류 발생: {e}")

# 3. 그래프 스타일 설정
plt.title('Battery Voltage Log (4 Arduinos)')
plt.xlabel('Time (