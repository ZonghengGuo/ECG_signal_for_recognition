import os

import pywt
from scipy.fft import fft
import numpy as np
from matplotlib import pyplot as plt
import tqdm

mode = 'CWT'  # Todo:[FFT, STFT, CWT]

# 先遍历所有文件夹下的npy文件并读取，求出频谱图
data_path = '/home/a/PycharmProjects/RR/databases/hamming window data/pulse'  # Todo: 输入路径
save_path = f"/home/a/PycharmProjects/RR/databases/preprocessing_{mode}/pulse"  # Todo: 输出路径
fs = 20

for sub_dir in tqdm.tqdm(os.listdir(data_path)):
    sub_dir_path = os.path.join(data_path, sub_dir)
    for sub_data_name in os.listdir(sub_dir_path):
        sub_data_path = os.path.join(sub_dir_path, sub_data_name)
        data = np.load(sub_data_path)

        # 生成频谱图
        if mode == 'FFT':
            fft_result = fft(data)

            # 计算频谱
            freq = np.fft.fftfreq(len(fft_result)) * fs
            magnitude = np.abs(fft_result)

            # 生成频谱图（伪彩图）
            plt.figure(figsize=(8, 6))
            plt.specgram(data, Fs=fs, NFFT=1024)

            # plt.show() # 如果要展示把这一行打开

            # 存储png文件
            basename, extension = os.path.splitext(sub_data_name)
            sub_save_path = save_path + '/' + sub_dir
            png_save_path = sub_save_path + '/' + basename + '.png'
            # print(sub_save_path)
            if not os.path.exists(sub_save_path):
                os.makedirs(sub_save_path)

            plt.savefig(png_save_path)

        elif mode == 'CWT':
            wavelet = 'db4'
            levels = 5
            coeffs = pywt.wavedec(data, wavelet, level=levels)

            reconstructed_wavelet = pywt.waverec(coeffs, wavelet)
            plt.specgram(reconstructed_wavelet, NFFT=256, Fs=2, noverlap=128, cmap='jet')
            plt.ylabel(u"Frequency (Hz)")
            plt.xlabel(u"Time (s)")
            plt.subplots_adjust(hspace=0.4)
            plt.title = ("CWT")
            # plt.show()

            # 存储png文件
            basename, extension = os.path.splitext(sub_data_name)
            sub_save_path = save_path + '/' + sub_dir
            png_save_path = sub_save_path + '/' + basename + '.png'
            # print(sub_save_path)
            if not os.path.exists(sub_save_path):
                os.makedirs(sub_save_path)

            plt.savefig(png_save_path)


