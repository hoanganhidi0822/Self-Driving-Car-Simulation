import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import albumentations as A
from albumentations.pytorch import ToTensorV2 
from PIL import Image

import os
from tqdm import tqdm
from glob import glob
                                        ###0-255####### 0-1 ###########-> tensor
class RoadDataset(Dataset): #transform: augmentation + norm + np.array -> torch.tensor
    
    color_encoding = [
        ('road', (31,120,180)),
        ('people', (227,26,28)),
        ('car', (106,61,154)),
        ('unlabeled', (0,0,0)),
        ]  
    
    def __init__(self, root_dir, txt_file, num_classes,transform=None): # khởi tạo 1 số thuộc tính, txt file
        super().__init__()
        
        self.num_classes = num_classes
        self.root_dir = root_dir
        self.txt_file = txt_file
        self.transform = transform
        self.img_path_lst = []

        with open(self.txt_file) as file_in:
            for line in file_in:
               img_path = line.strip()
               self.img_path_lst.append(img_path)
                
    
    def __len__(self):
        return len(self.img_path_lst)
    
    def convert_label_to_matrix(self, label):
        """
        Convert label image to matrix classes for applying cross-entropy loss.
        Return semantic index, label in enumemap of H x W x class.
        """
        semantic_map = np.zeros(label.shape[:-1])
        #Fill all value with 0 - defaul
        semantic_map.fill(self.num_classes - 1) #self.num_classes - 1
        #Fill the pixel with correct class
        for class_index, color_info in enumerate(self.color_encoding):
            color = np.array(color_info[1])
            equality = np.all(label == color, axis=-1)
            class_map = np.all(equality, axis=-1)
            semantic_map[class_map] = class_index

        return semantic_map
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, "images", f"{self.img_path_lst[idx]}.jpg")
        mask_path = os.path.join(self.root_dir, "labels", "{}.png".format(self.img_path_lst[idx]))
        
        image = np.array(Image.open(image_path).convert('RGB').resize((240, 320)))
        mask = np.array(Image.open(mask_path).convert('RGB').resize((240, 320)))
        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']
        else:
            transformed_image = image
            transformed_mask = mask
    
        # Convert label for cross-entropy
        transformed_mask = np.array(transformed_mask)
        transformed_mask = self.convert_label_to_matrix(transformed_mask)  # Pass 'num_classes' as a keyword argument
        transformed_mask = torch.from_numpy(transformed_mask).long()

        return transformed_image, transformed_mask
        
trainsize = 384

train_transform = A.Compose([
    A.Resize(width=trainsize, height=trainsize),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.Blur(),
    A.Sharpen(),
    A.RGBShift(),
    A.CoarseDropout(max_holes=5, max_height=25, max_width=25, fill_value=0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(),
])

test_trainsform = A.Compose([
    A.Resize(width=trainsize, height=trainsize),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
    ToTensorV2(), # numpy.array -> torch.tensor (B, 3, H, W)
])





