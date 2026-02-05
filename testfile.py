import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 설정 (가운데 5x5를 제외하고 None과 11x11만 추출)
x_labels = ['None', '11x11']
x = np.arange(len(x_labels))  # 라벨 위치 (0, 1)
width = 0.3  # 막대 두께

# ID Purity 데이터 (None, 11x11)
purity_scan = [32.7, 100.0]
purity_maxage = [31.8, 100.0]

# AssA 데이터 (None, 11x11)
assa_scan = [0.25, 0.76]
assa_maxage = [0.24, 0.58]

# ---------------------------------------------------------
# Figure 1: ID Purity (None vs 11x11)
# ---------------------------------------------------------
plt.figure(1, figsize=(6, 6))
plt.bar(x - width/2, purity_scan, width, label='Scan Topology Based', color='teal', alpha=0.8)
plt.bar(x + width/2, purity_maxage, width, label='Max Age = 5', color='orange', alpha=0.8)

plt.ylabel('ID Purity (%)', fontsize=12)
plt.xticks(x, x_labels)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend(loc='upper left')
plt.ylim(0, 120)

# 값 표시
for i in range(len(x)):
    plt.text(x[i] - width/2, purity_scan[i] + 2, f'{purity_scan[i]}%', ha='center', va='bottom', fontweight='bold')
    plt.text(x[i] + width/2, purity_maxage[i] + 2, f'{purity_maxage[i]}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('figure1_id_purity_comparison.png', dpi=300)

# ---------------------------------------------------------
# Figure 2: AssA (None vs 11x11)
# ---------------------------------------------------------
plt.figure(2, figsize=(6, 6))
plt.bar(x - width/2, assa_scan, width, label='Scan Topology Based', color='darkred', alpha=0.8)
plt.bar(x + width/2, assa_maxage, width, label='Max Age = 5', color='gray', alpha=0.8)

plt.ylabel('AssA Score', fontsize=12)
plt.xticks(x, x_labels)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend(loc='upper left')
plt.ylim(0, 1.0)

# 값 표시
for i in range(len(x)):
    plt.text(x[i] - width/2, assa_scan[i] + 0.02, f'{assa_scan[i]}', ha='center', va='bottom', fontweight='bold')
    plt.text(x[i] + width/2, assa_maxage[i] + 0.02, f'{assa_maxage[i]}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('figure2_assa_comparison.png', dpi=300)

plt.show()