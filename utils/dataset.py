from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from PIL import Image
import glob
import random
import numpy as np
import os
import cv2

class dataset(TensorDataset):
    def __init__(self, data_path_list, same_prob=0.8):
        datasets = []
        self.same_prob = same_prob
        for data_path in data_path_list:
            img_list = glob.glob(f'{data_path}/*.*g')
            datasets.append(img_list)

        self.datasets = sum(datasets, [])
        # print(datasets)
        
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        img_path = self.datasets[item]
        Xs = Image.open(img_path)

        if random.random() > self.same_prob:
            image_path = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])
            Xt = Image.open(img_path)
            same_person = 0
        else:
            Xt = Xs.copy()
            same_person = 1
        return self.transforms(Xs), self.transforms(Xt), same_person

    def __len__(self):
        return len(self.datasets)