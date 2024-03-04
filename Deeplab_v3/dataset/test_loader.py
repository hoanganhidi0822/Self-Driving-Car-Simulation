import os
from numpy.core.fromnumeric import searchsorted
import torch
from torch.utils import data
from tqdm import tqdm
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2 

from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
# from ex import BiSeNet
import numpy as np
from PIL import Image
from glob import glob
import random


class Image_loader(Dataset):
    color_encoding = [
        ('road', (31,120,180)),
        ('people', (227,26,28)),
        ('car', (106,61,154)),
        ('unlabeled', (0,0,0)),
        ]  

    def __init__(self, num_classes=4,mode='train'):
        self.num_classes = num_classes
        self.mode=mode
       
        #Normalization
        self.normalize = transforms.Compose([
            transforms.Resize((240,320)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ]) ##imagenet norm


        self.DATA_PATH =  '/media/hoanganh/New Volume/Documents/Researches/Self_Driving_Car/Deeplab_v3/dataset'
        """
        #print(self.DATA_PATH)
        self.train_path = self.DATA_PATH + '\data4k\images/'
        self.label_path = self.DATA_PATH + '\data4klabels/' """
        
        self.train_path = self.DATA_PATH + '/data4k/images/'
        self.label_path = self.DATA_PATH + '/data4k/labels/'
        #print(self.train_path)
        if self.mode == 'train':
            self.data_files = os.listdir(self.train_path)
            self.data_files.sort(key= lambda i: int(i.lstrip('data').rstrip('.jpg')))
            self.label_files = os.listdir(self.label_path)
            self.label_files.sort(key= lambda i: int(i.lstrip('data').rstrip('.png')))
            self.data_folder = []
            self.label_folder = []
            for f, datas in enumerate(self.data_files):
                file_data = self.train_path + self.data_files[f]
                file_label = self.label_path + self.label_files[f]
                self.data_folder.append(file_data)
                self.label_folder.append(file_label)

			
        else: 
            raise RuntimeError("Unexpected dataset mode."
                                "Supported modes: train")

    def __len__(self):
        return len(self.data_files)

    def get_files(self, data_folder):
        #
        return  glob("{}/*.{}".format(data_folder, 'jpg'))

    def get_label_file(self, data_folder):
        #
        return  glob("{}/*.{}".format(data_folder, 'png'))

    def image_loader(self, data_path, label_path):
        data = Image.open(data_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')

        
        return data, label.resize((320,240))

    def label_decode_cross_entropy(self, label):
        """
        Convert label image to matrix classes for apply cross entropy loss. 
        Return semantic index, label in enumemap of H x W x class
        """
        semantic_map = np.zeros(label.shape[:-1])
        #Fill all value with 0 - defaul
        semantic_map.fill(self.num_classes - 1) #self.num_classes - 1
        #Fill the pixel with correct class
        #print(semantic_map)
        for class_index, color_info in enumerate(self.color_encoding):
            color = np.array(color_info[1])
            # print(color.shape)
            # print(label.shape)
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            semantic_map[class_map] = class_index
            # print(semantic_map)
        return semantic_map

    def __getitem__(self, index):
        """
            Args:
            - index (``int``): index of the item in the dataset
            Returns:
            A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
            of the image.
        """
        data_path, label_path = self.data_folder[index], self.label_folder[index]
        img, label = self.image_loader(data_path, label_path)
        
        if self.mode == 'train' and random.random() > 0.5:
            # Horizontal flipping
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
         
        if self.mode == 'train' and random.random() > 0.8:
            rotation_degree = random.uniform(-10, 10)  # Adjust the range as needed
            img = img.rotate(rotation_degree, resample=Image.BICUBIC)
            label = label.rotate(rotation_degree, resample=Image.NEAREST)
            
        # img.show()
        # label.show()
        # Normalize image
        img = self.normalize(img)
        # Convert label for cross entropy
        label = np.array(label)
        label = self.label_decode_cross_entropy(label)
        # print(label)
        label = torch.from_numpy(label).long()
        return img, label
