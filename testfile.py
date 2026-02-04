import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 설정
x_labels = ['None', '5x5', '11x11']
x = np.arange(len(x_labels))  # 라벨 위치
width = 0.35  # 막대 두께

# ID Purity 데이터
purity_scan = [32.7, 75.5, 100.0]
purity_maxage = [31.8, 0, 100.0]  # 5x5 데이터가 없으므로 0으로 설정 (또는 시각적으로 비워둠)

# AssA 데이터
assa_scan = [0.25, 0.73, 0.76]
assa_maxage = [0.24, 0, 0.58]  # 5x5 데이터가 없으므로 0으로 설정

# ---------------------------------------------------------
# Figure 1: ID Purity (막대그래프)
# ---------------------------------------------------------
plt.figure(1, figsize=(8, 6))
plt.bar(x - width/2, purity_scan, width, label='Scan Topology Based', color='teal', alpha=0.8)
plt.bar(x + width/2, purity_maxage, width, label='Max Age = 5', color='orange', alpha=0.8)

plt.title('Figure 1: ID Purity Performance', fontsize=14, pad=15)
plt.xlabel('Grid Size', fontsize=12)
plt.ylabel('ID Purity (%)', fontsize=12)
plt.xticks(x, x_labels)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend()
plt.ylim(0, 115) # 상단 여유 공간

# 값 표시 (막대 위에 숫자 적기)
for i in range(len(x)):
    plt.text(x[i] - width/2, purity_scan[i] + 1, f'{purity_scan[i]}%', ha='center', va='bottom', fontsize=9)
    if purity_maxage[i] > 0:
        plt.text(x[i] + width/2, purity_maxage[i] + 1, f'{purity_maxage[i]}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figure1_id_purity_bar.png', dpi=300)

# ---------------------------------------------------------
# Figure 2: AssA (막대그래프)
# ---------------------------------------------------------
plt.figure(2, figsize=(8, 6))
plt.bar(x - width/2, assa_scan, width, label='Scan Topology Based', color='darkred', alpha=0.8)
plt.bar(x + width/2, assa_maxage, width, label='Max Age = 5', color='gray', alpha=0.8)

plt.title('Figure 2: AssA Performance', fontsize=14, pad=15)
plt.xlabel('Grid Size', fontsize=12)
plt.ylabel('AssA Score', fontsize=12)
plt.xticks(x, x_labels)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend()
plt.ylim(0, 1.0)

# 값 표시
for i in range(len(x)):
    plt.text(x[i] - width/2, assa_scan[i] + 0.02, f'{assa_scan[i]}', ha='center', va='bottom', fontsize=9)
    if assa_maxage[i] > 0:
        plt.text(x[i] + width/2, assa_maxage[i] + 0.02, f'{assa_maxage[i]}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figure2_assa_bar.png', dpi=300)

plt.show()