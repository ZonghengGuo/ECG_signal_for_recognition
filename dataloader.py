import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        """
        初始化函数
        :param root_dir: 数据集的根目录
        :param train: 一个布尔值，表示是否使用训练集
        :param transform: 一个可选的转换操作，用于对图像进行预处理
        """
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.data = []
        self.labels = []

        self._load_dataset()

    def _load_dataset(self):
        """
        加载数据集并进行处理
        """
        for label_dir in sorted(os.listdir(self.root_dir)):
            label_path = os.path.join(self.root_dir, label_dir)
            if os.path.isdir(label_path):
                images = sorted(os.listdir(label_path))
                N = len(images)
                clip_index = int(0.75*N)
                # 根据需求分割训练集和测试集
                if self.train:
                    images = images[:clip_index]
                else:
                    images = images[clip_index:]

                train_imgs = []
                test_imgs = []
                for img_name in images:
                    img_path = os.path.join(label_path, img_name)

                    if self.train:
                        train_imgs.append(img_name)
                    else:
                        test_imgs.append(img_name)
                    self.data.append(img_path)
                    self.labels.append(int(label_dir))
                # print("train_imgs", train_imgs)
                # print("test_imgs", test_imgs)

    def __len__(self):
        """
        返回数据集中的样本数
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据给定的索引idx返回一个样本
        """
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')  # 转换为RGB格式
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label


