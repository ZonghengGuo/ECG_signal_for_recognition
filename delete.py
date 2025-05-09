import os
import math
import shutil

# 设置要操作的目标文件夹路径
target_folder = "/root/autodl-tmp/RR/databases/crop_data/self_made/0.5/9"

# 获取目标文件夹中文件的列表
files = os.listdir(target_folder)

# 计算要保留的文件数，即总数的0.2倍
keep_files = math.ceil(len(files) * 0.2)

# 按照文件的修改时间对文件进行排序
files.sort(key=lambda x: os.path.getmtime(os.path.join(target_folder, x)))

# 删除多余的文件
files_to_delete = files[:-keep_files]
for file_name in files_to_delete:
    file_path = os.path.join(target_folder, file_name)
    if os.path.isfile(file_path):
        os.remove(file_path)
    elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
