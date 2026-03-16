import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 1. 불러올 파일 이름 리스트
file_names = ["1.csv", "2.csv", "3.csv", "4.csv"]

# 2x2 서브플롯 생성
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, file in enumerate(file_names):
    ax = axes[i]
    try:
        # 데이터 로드
        df = pd.read_csv(file, header=0, on_bad_lines='skip')
        
        # 숫자 변환 및 결측치 제거
        df['time_s'] = pd.to_numeric(df['time_s'], errors='coerce')
        df['vbat_mV'] = pd.to_numeric(df['vbat_mV'], errors='coerce')
        df = df.dropna(subset=['time_s', 'vbat_mV'])
        
        # [반영] 앞뒤 10개 데이터 제외
        if len(df) > 20:
            df_filtered = df.iloc[10:-10].copy()
        else:
            df_filtered = df.copy()
            
        if not df_filtered.empty:
            # 그래프 그리기
            ax.plot(df_filtered['time_s'], df_filtered['vbat_mV'], color='tab:blue', linewidth=1.2)
            
            # [반영] Y축 눈금을 5단위로 설정
            ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
            
            ax.set_title(f'Target {i+1} ({file})',fontsize=7)
            ax.set_xlabel('Time (Seconds)', fontsize=8)
            ax.set_ylabel('Voltage (mV)', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # 눈금이 많아질 수 있으므로 폰트 크기 조절
            ax.tick_params(axis='y', labelsize=7)

    except Exception as e:
        ax.set_title(f'⚠️ Error in {file}')

plt.suptitle('Battery Voltage', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()