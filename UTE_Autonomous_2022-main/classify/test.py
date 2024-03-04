# import torch.nn as nn
import cv2
from PIL import Image
import numpy as np
from classify.model import Network
import torch
import matplotlib.pyplot as plt
import os 
import time
classes = ['dithang', 'retrai', 'rephai', 'noleft', 'noright', 'nostraight']
device = torch.device("cuda")
img_size = (64, 64)
model = Network()
pre_trained = torch.load("./classify/traffic_sign.pth")
model.load_state_dict(pre_trained)
model = model.to(device)
model.eval()

def Predict(img_raw):

    img_rgb = cv2.resize(img_raw, img_size)
   
    # convert from RGB img to gray img
    # img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    # normalize img from [0, 255] to [0, 1]
    img_rgb = img_rgb/255
    img_rgb = img_rgb.astype('float32')
    img_rgb = img_rgb.transpose(2,0,1)
   

    # convert image to torch with size (1, 1, 48, 48)
    img_rgb = torch.from_numpy(img_rgb).unsqueeze(0)

    with torch.no_grad():
        img_rgb = img_rgb.to(device)
        # print("type: " + str(numb), type(img_rgb))
        y_pred = model(img_rgb)
          
        _, pred = torch.max(y_pred, 1)
        
        pred = pred.data.cpu().numpy()
        # print("2nd", second_time - fist_time)
        # print("predict: " +str(numb), pred)
        class_pred = classes[pred[0]]
        # print("class_pred: ", class_pred)
        
        
    return class_pred

# dithang:  99.08116385911178
# rephai: 99.64114832535886
# retrai: 99.20127795527156
# noleft: 98.25783972125436
# noright: 98.95833333333334
# nostraight: 98.92904953145917

# img_name = os.listdir("./classify/data/nostraight/")
# print(img_name[0])
# i = 0
# for name in img_name:
#     img = cv2.imread("./classify/data/nostraight/" + name)
#     result = Predict(img)
#     if result != "nostraight":
#         i += 1
#         # print(name, result, i)

# print("correct(%) : ", 100*(1 - (i/len(img_name))))





