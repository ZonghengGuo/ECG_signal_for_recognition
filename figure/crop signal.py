import os
import numpy as np
from matplotlib import pyplot as plt
import h5py

# 找出最短的信号，将其长度作为baseline
# 信号最短的是1920 (96秒) 可以预估16s为一个窗
data_path = 'E:/datasets/COHFACE'
save_path = 'E:/code/RR/databases/crop_data/pulse'
lenth = []

# 读取COHFACE数据集总共４０个人，每个人有４个pulse波，读取并查看长度
for sub_dir in os.listdir(data_path):
    sub_dir_path = os.path.join(data_path, sub_dir)
    i=0
    for sub_sub_dir in os.listdir(sub_dir_path):
        sub_sub_dir_path = os.path.join(sub_dir_path, sub_sub_dir)
        sub_sub_data_path = sub_sub_dir_path + '/data.hdf5'
        with h5py.File(sub_sub_data_path, 'r') as f:
            data = f['pulse'][20:15020]

        # lenth.append(len(data))

        save_sub_path = os.path.join(save_path, sub_dir)
        if not os.path.exists(save_sub_path):
            os.makedirs(save_sub_path)
        save_sub_name = save_sub_path + f"/data_{i}.npy"
        print(save_sub_name)
        np.save(save_sub_name, data)
        i += 1

# 读取自制数据集
#     for sub_data_name in os.listdir(sub_dir_path):
#         sub_data_path = os.path.join(sub_dir_path, sub_data_name)
#         with open(sub_data_path, 'rb') as file:
#             data = file.read()
#         data = np.frombuffer(data, dtype=np.uint8)
# #         lenth.append(len(data))
# # print(min(lenth))
#
# # 裁剪信号为1920的长度，并保存到crop_data/breath文件夹下
#         data = data[:1920]
#         print(len(data))
#
#         save_sub_path = os.path.join(save_path, sub_dir)
#         if not os.path.exists(save_sub_path):
#             os.makedirs(save_sub_path)
#         save_sub_name = save_sub_path + "/data.npy"
#         print(save_sub_name)
#         np.save(save_sub_name, data)
#
# # 验证读取一遍
# for sub_dir in os.listdir(save_path):
#     sub_dir_path = os.path.join(save_path, sub_dir)
#     val_data = sub_dir_path + "/data.npy"
#     val_data = np.load(val_data)
#     print(len(val_data))
