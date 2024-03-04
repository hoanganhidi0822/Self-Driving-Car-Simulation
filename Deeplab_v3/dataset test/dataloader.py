import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
""" import torchmetrics
from torchmetrics import Dice, JaccardIndex
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2 # np.array -> torch.tensor """
import os
from tqdm import tqdm
from glob import glob
                                        ###0-255####### 0-1 ###########-> tensor
class RoadDataset(Dataset): #transform: augmentation + norm + np.array -> torch.tensor
    def __init__(self, root_dir, txt_file, transform=None): # khởi tạo 1 số thuộc tính, txt file
        super().__init__()
        self.root_dir = root_dir
        self.txt_file = txt_file
        self.transform = transform
        self.img_path_lst = []
        with open(self.txt_file) as file_in:
            for line in file_in:
                self.img_path_lst.append(line.split(" ")[0])
                
            print(len(self.img_path_lst))   
    
    def __len__(self):
        return len(self.img_path_lst)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, "images", "{}.jpg".format(self.img_path_lst[idx]))
        mask_path = os.path.join(self.root_dir, "annotations", "trimaps", "{}.png".format(self.img_path_lst[idx]))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # foreground -> 1
        # background 2 -> 0
        # 3 -> 1
        mask[mask == 2] = 0
        mask[mask == 3] = 1
        # image (RGB), mask (2D matrix)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']
        return transformed_image, transformed_mask


trainsize = 384

train_transform = A.Compose([
    A.Resize(width=trainsize, height=trainsize),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.Blur(),
    A.Sharpen(),
    A.RGBShift(),
    A.Cutout(num_holes=5, max_h_size=25, max_w_size=25, fill_value=0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
])

test_trainsform = A.Compose([
    A.Resize(width=trainsize, height=trainsize),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(), # numpy.array -> torch.tensor (B, 3, H, W)
])

 
traindata = RoadDataset('D:\Documents\Researches\Self_Driving_Car\Deeplab_v3\dataDogCat\images', 'D:\Documents\Researches\Self_Driving_Car\Deeplab_v3\dataDogCat/annotations/annotations/trainval.txt')
traindata.__getitem__(0)


image = cv2.imread('D:\Documents\Researches\Self_Driving_Car\Deeplab_v3\dataDogCat\images\images\Abyssinian_100.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(np.real(image))
plt.show()


