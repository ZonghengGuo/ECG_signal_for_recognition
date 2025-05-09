import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter

data_path = "/home/a/PycharmProjects/RR/databases/crop_data/pulse"  # Todo：输入路径
save_path = "/home/a/PycharmProjects/RR/databases/hamming window data/pulse"  # Todo：输出路径

# 对信号裁剪，裁剪长度为T,然后控制重叠部分为T/2
window_length = 1250  # Todo:设置窗口长度
overlapping_length = 625  # Todo：设置重叠长度
step = window_length - overlapping_length

for sub_name in os.listdir(data_path):
    sub_data_path = os.path.join(data_path, sub_name)
    i = 0  # 用作存储的索引
    for file_name in os.listdir(sub_data_path):
        data = sub_data_path + '/' + file_name
        data = np.load(data)
        N = len(data)

        # 裁剪取出窗口，会生成Ｎ×２－１个窗口
        windows = []
        end = window_length
        for start in range(0, N - window_length + 1, step):
            end = start + window_length
            window = data[start: end]
            windows.append(window)

            i += 1
            sub_save_path = os.path.join(save_path, sub_name)
            if not os.path.exists(sub_save_path):
                os.makedirs(sub_save_path)
            window_save_path = sub_save_path + f"/{i}.npy"
            np.save(window_save_path, window)

            # 验证一下是否为T
            val_data = np.load(window_save_path)
            print(len(val_data))




