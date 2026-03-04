import pandas as pd
import matplotlib.pyplot as plt

# 1. 불러올 파일 이름 리스트 (1~4번 아두이노)
file_names = [
    "battery_log_1.csv", 
    "battery_log_2.csv", 
    "battery_log_3.csv", 
    "battery_log_4.csv"
]

plt.figure(figsize=(10, 6))

for i, file in enumerate(file_names):
    try:
        # 아두이노가 첫 줄에 'idx,time_s,vbat_mV,percent' 헤더를 던져주므로 header=0 사용
        # on_bad_lines='skip'을 통해 중간에 섞인 이상한 에러 문자 무시
        df = pd.read_csv(file, header=0, on_bad_lines='skip')
        
        # 파일 맨 끝에 있는 "END" 문자열이나, 중간에 섞일 수 있는 "[LOG] Vbat..." 문자열 처리
        # 숫자로 변환 안 되는 값들은 NaN으로 강제 변환 후 제거
        df['time_s'] = pd.to_numeric(df['time_s'], errors='coerce')
        df['vbat_mV'] = pd.to_numeric(df['vbat_mV'], errors='coerce')
        df = df.dropna(subset=['time_s', 'vbat_mV'])
        
        # X축: 시간(초), Y축: 배터리 전압(mV)
        plt.plot(df['time_s'], df['vbat_mV'], label=f'Arduino {i+1}', linewidth=1.5)

    except FileNotFoundError:
        print(f"⚠️ {file} 파일이 없습니다.")
    except Exception as e:
        print(f"⚠️ {file} 처리 중 오류: {e}")

# 3. 그래프 스타일 설정
plt.title('Battery Voltage Over Time (4 Arduinos)')
plt.xlabel('Time (Seconds)')
plt.ylabel('Voltage (mV)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

plt.show()