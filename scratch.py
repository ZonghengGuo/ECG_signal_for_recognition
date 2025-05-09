import scipy.io as scio
from matplotlib import pyplot as plt

data_path = 'databases/raw/ecg_for_30/0/GDN0001_1_Resting.mat'

data = scio.loadmat(data_path)
data1 = data['tfm_ecg1']
fs = data['fs_ecg'][0][0]
ecg1 = []
print(len(data1))
time = len(data1) // fs
print(time)
data1 = data1[: 30 * fs]
print(fs)
for i in data1:
    temp = i[0]
    ecg1.append(temp)

plt.plot(ecg1)
plt.show()
threshold = 1
window_length = int(fs * threshold)


