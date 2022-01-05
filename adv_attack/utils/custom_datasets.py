import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class Dataset_Cifar10(Dataset):
    def __init__(self, labelfile_path, transform=None):
        self.image_path_list, self.image_label_list = [], []
        self.image_list = []
        self.transform = transform

        labelfile = open(labelfile_path, 'r')
        for line in labelfile:
            line = line[:-1]
            infos = line.split(' ')
            image_path = infos[0]
            image_label = infos[1]
                
            # image_path = image_name

            if os.path.exists(image_path):
                self.image_path_list.append(image_path)
                self.image_label_list.append(int(image_label))
        
        for index in range(len(self.image_path_list)):
            image_path = self.image_path_list[index]
            image = Image.open(image_path)
            image_np = np.asarray(image)
            image_np = image_np.transpose(2, 0, 1)
            image_np = image_np / 255
            self.image_list.append(image_np)

        self.image_list_np = np.array(self.image_list)
        self.image_label_list_np = np.array(self.image_label_list)
        
        print(self.image_list_np.shape, self.image_list_np.dtype)
        print(self.image_label_list_np.shape, self.image_label_list_np.dtype)
    
    def __getitem__(self, index):
        image, label = self.image_list_np[index], self.image_label_list_np[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.image_list_np.shape[0]