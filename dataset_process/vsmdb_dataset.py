import librosa
from scipy import signal

import scipy.io as sio
import os
import pywt
import numpy as np
from matplotlib import pyplot as plt
import tqdm
import librosa.display


def crop_window(dataset_path, T, save_path, threshold):
    print("Start cropping signal.....")
    for person in tqdm.tqdm(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, person)
        index = 0  # 用作存储的索引
        for mat_file in os.listdir(person_path):
            mat_path = os.path.join(person_path, mat_file)
            data = sio.loadmat(mat_path)  # Fs, ecg_lead2, pcg_audio
            fs = data['Fs'][0][0]
            ecg = data['ecg_lead2']
            # pcg = data['pcg_audio']

            # cache_pcg = []
            cache_ecg = []
            for i in range(len(ecg)):
                # cache_pcg.append(pcg[i][0])
                cache_ecg.append(ecg[i][0])
            ecg_data = cache_ecg[:(T * fs)]
            # pcg_data = cache_pcg[:(T*fs)]

            # 对ecg信号做滤波，频段范围为0.5-45Hz
            b_ecg, a_ecg = signal.butter(4, [0.5 / fs * 2, 100 / fs * 2], 'bandpass')
            ecg_data = signal.filtfilt(b_ecg, a_ecg, ecg_data)

            # 配置参数
            window_length = int(fs * threshold)  # Todo:修改窗口长度，fs=1s
            overlapping = int(0.5 * window_length)
            step = window_length - overlapping
            n = len(ecg_data)

            windows = []
            end = window_length
            for start in range(0, n - window_length + 1, step):
                end = start + window_length
                window = ecg_data[start: end]
                windows.append(window)

                index += 1
                sub_save_path = os.path.join(save_path, person)
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
                spect = librosa.feature.melspectrogram(y=data, sr=2000, n_fft=512, hop_length=16, win_length=512)
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
    dataset = 'vsmdb'
    window_rate = 0.5
    time = 30 * window_rate * 2  # 裁剪时间(s)
    spec_mode = 'FFT'  # Todo:预处理模式 ['FFT', 'STFT', 'CWT', 'Mel']
    data_path = f'databases/raw/{dataset}'
    crop_output_path = f'databases/crop_data/{dataset}/{window_rate}'
    spec_output_path = f'databases/preprocess_spectrogram/preprocessing_{spec_mode}/{dataset}/{window_rate}'

    crop_window(data_path, time, crop_output_path, window_rate)
    spectrogram_generator(crop_output_path, spec_output_path, spec_mode)