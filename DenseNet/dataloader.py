import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import read_image

class CustomDataset(Dataset):
    # image_dir为数据目录，label_file，为标签文件
    def __init__(self, image_file, label_file, transform=None):
        self.image_file = pd.read_csv(image_file)
        self.label_file = pd.read_csv(label_file) 
        self.transform = transform 
    
    # 加载每一项数据
    def __getitem__(self, idx):
        # 每个图片，其中idx为数据索引
        img_name = os.path.join(self.image_dir, '%.3d.jpg' % (idx + 1)) # 加载每一张照片
        image = read_image(img_name)

        # 对应标签
        labels = self.label_file['labels'][idx]

        if self.transform:
            image = self.transform(image)

        # 返回一张照片，一个标签
        return image, labels
    
    # 数据集大小
    def __len__(self):
        return (len(self.label_file))