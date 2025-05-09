import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, welch
import h5py
import scipy


# 测试，对某一呼吸波形滤波并通过PSD计算出呼吸率
input_file = 'E:\\datasets\\COHFACE\\1\\1\\data.hdf5'

with h5py.File(input_file, 'r') as f:
    data = f['pulse'][1000:2000]

# with open(input_file, 'rb') as file:
#     data = file.read()

# 二进制读取
# data = np.load(input_file)
# t = np.linspace(0, len(data), len(data))

fs = 20
# butter滤波器滤波
b, a = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
filtered_data = scipy.signal.filtfilt(b, a, np.double(data))
time = np.arange(len(data)) / fs / 10

# PSD
frequencies, powers = welch(data, fs, nperseg=1000)
mask = (frequencies >= 0.1) & (frequencies <= 2.5)
breathing_freq_idx = np.argmax(powers[mask])
breathing_freq = frequencies[mask][breathing_freq_idx]

# print(f"Estimated breathing frequency: {breathing_freq * 60} Hz")

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(time, data)
plt.title('(a) Original signal', y =-0.4)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Signal')
ax1.grid()
# ax1.set_xlim(0, 50)

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(time, filtered_data)
plt.title('(b) Filtered signal', y =-0.4)
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('')
ax2.grid()
# ax2.set_xlim(0, 50)

breathing_freq = 1.12
frequencies = frequencies + 1
# print(frequencies)
temp1 = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
temp2 = [0, 0, 0, 0, 0, 0]
frequencies = np.concatenate((temp1, frequencies))
powers = np.concatenate((temp2, powers))


ax3 = fig.add_subplot(2, 1, 2)
ax3.plot(frequencies, powers)
plt.title('(c) PSD', y =-0.4)
ax3.set_xlabel('Frequency [Hz]')
ax3.set_ylabel('Power')
ax3.axvline(x=float(breathing_freq), color='r', linestyle='-', label=f'Estimated Breathing Frequency: {breathing_freq:.2f} Hz')
ax3.set_xlim(0, 2)
ax3.grid()

# Adjust the layout so everything fits
fig.tight_layout()
plt.savefig('Output/Res_signal_propose')

# Show the plot
plt.show()
