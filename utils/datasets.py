import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import sys


import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import rotate, zoom

class DataAugmentation:
    def __init__(self, output_size):
        self.output_size = output_size

    def random_hue_saturation_value(self, image, hue_shift_limit=(-30, 30),
                                    sat_shift_limit=(-5, 5),
                                    val_shift_limit=(-15, 15)):
        if np.random.random() < 0.5:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(image)
            hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
            sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
            val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])

            h = cv2.add(h, hue_shift)
            s = cv2.add(s, sat_shift)
            v = cv2.add(v, val_shift)

            image = cv2.merge((h, s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        return image

    def random_shift_scale_rotate(self, image, shift_limit=(-0.1, 0.1),
                                  scale_limit=(-0.1, 0.1),
                                  rotate_limit=(-90, 90)):
        if np.random.random() < 0.5:
            height, width, channels = image.shape

            angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
            scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
            dx = np.random.uniform(shift_limit[0], shift_limit[1]) * width
            dy = np.random.uniform(shift_limit[0], shift_limit[1]) * height

            cc = np.cos(angle/180*np.pi) * scale
            ss = np.sin(angle/180*np.pi) * scale
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])
            corners = corners - np.array([width/2, height/2])
            corners = np.dot(corners, rotate_matrix)
            corners = corners + np.array([width/2 + dx, height/2 + dy])

            mat = cv2.getAffineTransform(np.float32([corners[0], corners[1], corners[3]]), 
                                         np.float32([[0, 0], [width, 0], [0, height]]))
            image = cv2.warpAffine(image, mat, (width, height))

        return image

    def random_flip(self, image):
        if np.random.random() < 0.5:  # horizontal flip
            image = cv2.flip(image, 1)
        if np.random.random() < 0.5:  # vertical flip
            image = cv2.flip(image, 0)
        return image

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # Convert to RGB (opencv loads images in BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = self.random_hue_saturation_value(image)
        image = self.random_shift_scale_rotate(image)
        image = self.random_flip(image)

        # Resize if needed
        if image.shape[:2] != self.output_size:
            image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_AREA)
            label = cv2.resize(label, self.output_size, interpolation=cv2.INTER_NEAREST)

        # Normalize and convert to tensor
        image = torch.from_numpy(image / 255.0).float().permute(2, 0, 1)  # CHW format
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}
    


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=1, reshape=False)  # 使用order=1，双线性插值
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y, _ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=1)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32))
        image = image.permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class load_dataset_ship(Dataset):
    def __init__(self, base_dir, transform=None):
        self.transform = transform
        self.data_paths = [os.path.join(base_dir, fname) for fname in os.listdir(base_dir) if fname.endswith('.npz')]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        data = np.load(data_path)
        image, label = data['image'], data['label']

        # 仅保留标签为3的像素点，其他像素点置为0
        label = (label == 3).astype(np.int32)

        # 对于训练数据的特殊处理
        if self.transform:
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
            return sample
        
        # 如果没有特殊处理，直接返回处理后的图像和标签
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # 从 HWC 转换为 CHW
        label = torch.from_numpy(label.astype(np.int32)).long()  # 保证标签为整数型
        
        return {'image': image, 'label': label, 'case_name': os.path.basename(data_path)}


class load_dataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.transform = transform
        self.data_paths = [os.path.join(base_dir, fname) for fname in os.listdir(base_dir) if fname.endswith('.npz')]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        data = np.load(data_path)
        image, label = data['image'], data['label']

        # 对于训练数据的特殊处理
        if self.transform:
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
            return sample
        
        # 如果没有特殊处理，直接返回处理后的图像和标签
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # 从 HWC 转换为 CHW
        label = torch.from_numpy(label.astype(np.float32)).long()
        
        return {'image': image, 'label': label, 'case_name': os.path.basename(data_path)}

# 现在，你可以像这样创建一个数据集实例：
# dataset = load_dataset(base_dir='你的数据目录路径', transform=RandomGenerator(output_size=(128, 128)))


class load_dataset_old(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name)
            data = np.load(data_path)
            image, label = data['image'], data['label']
            
        else:
            
            slice_name = self.sample_list[idx].strip('\n')
            data_path = self.data_dir+"/"+slice_name
            data = np.load(data_path)
            image, label = data['image'], data['label']
            image = torch.from_numpy(image.astype(np.float32))
            image = image.permute(2,0,1)
            label = torch.from_numpy(label.astype(np.float32))
        
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
