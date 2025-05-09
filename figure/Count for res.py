import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter, welch

# 先遍历所有文件夹下的dat文件并读取，计算出5秒内窗口下的RR，存储在列表中
data_path = '../databases/raw/DIY/breath'
fs = 20

window_size = fs * 20  # 频率 × 窗口时间长度
res_freq_list = []

for sub_dir in os.listdir(data_path):
    sub_dir_path = os.path.join(data_path, sub_dir)
    for sub_data_name in os.listdir(sub_dir_path):
        print(sub_data_name)
        sub_data_path = os.path.join(sub_dir_path, sub_data_name)

        # 读取每个文件夹爱下的data
        with open(sub_data_path, 'rb') as file:
            data = file.read()
        data = np.frombuffer(data, dtype=np.uint8)

        # 对每个data滤波
        b, a = butter(1, [0.1 / 20 * 2, 0.5 / 20 * 2], btype='bandpass')
        filtered_data = lfilter(b, a, np.double(data))

        # 加窗,PSD并存储
        for i in range(len(data) // window_size):
            window_data = filtered_data[i*window_size:(i+1)*window_size]
            # print(len(window_data))

            # 对窗口data做PSD
            frequencies, powers = welch(window_data, fs, nperseg=400)
            mask = (frequencies >= 0.1) & (frequencies <= 0.5)
            if len(powers[mask]) == 0:
                continue
            breathing_freq_idx = np.argmax(powers[mask])
            breathing_freq = frequencies[mask][breathing_freq_idx]
            breathing_count = breathing_freq * 60

            print(f"Estimated breathing frequency: {breathing_count}")  # 平均呼吸率
            res_freq_list.append(breathing_count)

# 绘制直方图
plt.hist(res_freq_list, bins='auto', alpha=0.7, rwidth=0.85)

# 设置标题和轴标签
plt.title('Breathing Rate Histogram')
plt.xlabel('Breathing Rate')
plt.ylabel('Frequency')

# 显示图表
plt.show()


