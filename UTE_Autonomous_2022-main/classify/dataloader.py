import os
from torch.utils.data import Dataset
import pandas as pd
import cv2
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

class Image_Loader(Dataset):
    def __init__(self, root_path='./data_train.csv', image_size=[48, 48], transforms_data=True):
        
        self.data_path = pd.read_csv(root_path)
        self.image_size = image_size
        self.num_images = len(self.data_path)
        self.transforms_data = transforms_data
        
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):

        # read the images from image path
        image_path = os.path.join(self.data_path.iloc[idx, 0])
        image = cv2.imread(image_path)
        augment_hsv(image, 0, 0.5, 0.5)
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        # image = Image.open(image_path)
        
        # read label
        label_cross = self.data_path.iloc[idx, 1]

        if self.transforms_data == True:
            data_transform = self.transform(True, True, False)
            image = data_transform(image)

        return image, torch.from_numpy(np.array(label_cross, dtype=np.long))#, image_path

    def transform(self, resize, totensor, normalize):
        options = []

        # if flip:
        #     print('oh nooooooooooooo')
        #     options.append(transforms.RandomHorizontalFlip())
        if resize:
            options.append(transforms.Resize(self.image_size))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.1307,), (0.3081,)))
        
        transform = transforms.Compose(options)

        return transform

def augment_hsv(im, hgain= 0, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=im)  # no return needed