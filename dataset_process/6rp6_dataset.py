import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from scipy import signal
import tqdm
import librosa
import pywt

def read_csv_then_crop(input_path, save_path, threshold, time):
    fs = 1000
    print("Start cropping signal.....")
    for csv_file_path in tqdm.tqdm(os.listdir(input_path)):
        index = 0  # 用作存储的索引
        csv_file = pd.read_csv(os.path.join(input_path, csv_file_path), low_memory=False)
        ecg = (csv_file.iloc[22:, 5].values)[120 * fs : (120+int(time)) * fs]
        ecg_data = [float(item) for item in ecg]

        # 对ecg信号做滤波，频段范围为0.5-45Hz
        b_ecg, a_ecg = signal.butter(4, [0.5 / fs * 2, 100 / fs * 2], 'bandpass')
        ecg_data = signal.filtfilt(b_ecg, a_ecg, ecg_data)

        # 配置参数
        window_length = int(fs * threshold)  # Todo:修改窗口长度，fs=1s
        overlapping = int(0.8 * window_length)
        step = window_length - overlapping
        n = len(ecg_data)

        windows = []
        end = window_length
        for start in range(0, n - window_length + 1, step):
            end = start + window_length
            window = ecg_data[start: end]
            windows.append(window)

            index += 1
            number = csv_file_path[0]
            sub_save_path = os.path.join(save_path, number)
            if not os.path.exists(sub_save_path):
                os.makedirs(sub_save_path)
            window_save_path = sub_save_path + f"/{index}.npy"
            np.save(window_save_path, window)


def spectrogram_generator(dataset_path, save_path, mode):
    print("\n")
    print("Start generating spectrogram.....")
    for sub_dir in tqdm.tqdm(os.listdir(dataset_path)):
        sub_dir_path = os.path.join(dataset_path, sub_dir)
        for sub_data_name in os.listdir(sub_dir_path):
            sub_data_path = os.path.join(sub_dir_path, sub_data_name)
            data = np.load(sub_data_path)
            fs = len(data)

            # 生成频谱图
            if mode == 'FFT':
                # 生成频谱图（伪彩图）
                plt.figure(figsize=(3, 2))
                plt.specgram(data, Fs=4096, NFFT=128, noverlap=64)

                # plt.show() # 如果要展示把这一行打开

                # 存储png文件
                basename, extension = os.path.splitext(sub_data_name)
                sub_save_path = save_path + '/' + sub_dir
                png_save_path = sub_save_path + '/' + basename + '.png'
                # print(sub_save_path)
                if not os.path.exists(sub_save_path):
                    os.makedirs(sub_save_path)

                plt.savefig(png_save_path)
                plt.close()

            elif mode == 'CWT':
                wavelet = 'db4'
                levels = 4
                coeffs = pywt.wavedec(data, wavelet, level=levels)

                reconstructed_wavelet = pywt.waverec(coeffs, wavelet)
                plt.figure(figsize=(3, 2))
                plt.specgram(reconstructed_wavelet, NFFT=128, Fs=4096, noverlap=64, cmap='jet')
                plt.ylabel(u"Frequency (Hz)")
                plt.xlabel(u"Time (s)")
                plt.subplots_adjust(hspace=0.4)
                plt.title = ("DWT")
                # plt.show()

                # 存储png文件
                basename, extension = os.path.splitext(sub_data_name)
                sub_save_path = save_path + '/' + sub_dir
                png_save_path = sub_save_path + '/' + basename + '.png'
                if not os.path.exists(sub_save_path):
                    os.makedirs(sub_save_path)

                plt.savefig(png_save_path)
                plt.close()
            elif mode == 'Mel':
                plt.figure(figsize=(3, 2))
                spect = librosa.feature.melspectrogram(y=data, sr=1000, n_fft=512, hop_length=16, win_length=512)
                mel_spect = librosa.power_to_db(spect, ref=np.max)
                librosa.display.specshow(mel_spect, sr=fs, x_axis='time', y_axis='mel')

                plt.ylabel('Mel Frequency')
                plt.xlabel('Time(s)')
                plt.title('Mel Spectrogram')
                # plt.show()

                # 存储png文件
                basename, extension = os.path.splitext(sub_data_name)
                sub_save_path = save_path + '/' + sub_dir
                png_save_path = sub_save_path + '/' + basename + '.png'
                if not os.path.exists(sub_save_path):
                    os.makedirs(sub_save_path)

                plt.savefig(png_save_path)
                plt.close()

if __name__ == '__main__':
    window_rate = 0.5
    overlapping = 0.8
    dataset = '6rp6'
    spec_mode = 'Mel'
    time_length = 30 * window_rate * 2  # 裁剪时间(s)
    dataset_path = 'databases/raw/6rp6wrd2pr-2/Data set of radar signal (.csv)'
    crop_save_path = f'databases/crop_data/{dataset}/{overlapping}'
    spec_output_path = f'databases/preprocess_spectrogram/preprocessing_{spec_mode}/{dataset}/{overlapping}'

    read_csv_then_crop(dataset_path, crop_save_path, window_rate, time_length)
    spectrogram_generator(crop_save_path, spec_output_path, spec_mode)