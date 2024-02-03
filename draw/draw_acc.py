import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 数据
session_lengths = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
                            160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
                            325, 350, 375, 400, 425, 450, 475, 500, 550, 600, 650, 700, 750, 800, 850,
                            900, 950, 1000, 1010])

accuracies = np.array([0.6331, 0.6425, 0.6491, 0.6511, 0.6509, 0.6526, 0.6568, 0.6583, 0.6556, 0.6547, 0.6538, 0.6523, 0.6514, 0.6494,
                       0.6524, 0.6511, 0.6509, 0.6489, 0.6506, 0.6493, 0.6468, 0.6477, 0.6452, 0.6422, 0.6432, 0.6421, 0.6421, 0.6407, 0.6410,
                       0.6405, 0.6395, 0.6398, 0.6404, 0.6402, 0.6398, 0.6399, 0.6394, 0.6391, 0.6387, 0.6390, 0.6384, 0.6388, 0.6390, 0.6386,
                       0.6389, 0.6389, 0.6389, 0.6389])

# 平滑曲线
spl = make_interp_spline(session_lengths, accuracies, k=3)
session_lengths_smooth = np.linspace(session_lengths.min(), session_lengths.max(), 300)
accuracies_smooth = spl(session_lengths_smooth)

# 画图
plt.figure(figsize=(8, 6))
plt.plot(session_lengths_smooth, accuracies_smooth)#, label="Secondary Structure ACC")
# plt.scatter(session_lengths, accuracies, color='red', marker='o', label='Data Points')
plt.title('Secondary Structure Accuracy over different Session Length')
plt.xlabel('Session Length')
plt.ylabel('Accuracy')
# plt.legend()
plt.grid(True)
plt.show()

plt.savefig('./acc.png')
