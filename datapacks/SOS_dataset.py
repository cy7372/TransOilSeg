import torch
from torch.utils.data import Dataset
from torchvision import transforms
from dancher_tools.utils.data_loader import DatasetRegistry
from PIL import Image
import os
import numpy as np

@DatasetRegistry.register_dataset('SOS_dataset')
class SOS_dataset(Dataset):
    """
    SOS 数据分割数据集类，假设图像和掩码已在 data_loader 中预处理。
    只有黑色（0）和白色（1）两类，使用颜色映射。
    """
    color_map = {
        (0, 0, 0): 0,  # 黑色 - 背景（0）
        (255, 255, 255): 1  # 白色 - 前景（1）
    }
    
    class_name = ["background", "oil spill"]
    def __init__(self, data, transform=None):
        """
        初始化数据集类。
        :param data: 缓存数据，包含 'images' 和 'masks'
        :param transform: 图像变换，默认为 None
        """
        self.images = data['images']  # 已预处理的图像
        self.masks = data['masks']  # 已转换为类别索引的掩码
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        """
        返回数据集的大小。
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        获取数据集中的单个样本，包括图像和掩码。
        """
        # 从缓存中直接获取图像和掩码
        image = self.images[idx]  # 已是预处理好的 NumPy 格式
        mask = self.masks[idx]    # 已是类别索引的掩码

        # 将 NumPy 格式图像应用转换
        image = self.transform(image)

        return image, torch.tensor(mask, dtype=torch.long)