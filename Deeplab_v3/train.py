
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm 
#from model.deeplab_new import DeepLabV3
from model.deeplabv3 import DeepLabV3
#from dataset import dataloader

from dataset import dataloader
import albumentations as A
from albumentations.pytorch import ToTensorV2 
#import yaml
from addict import Dict
import argparse
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from helpers import (
    reverse_one_hot,
    compute_accuracy,
    fast_hist,
    per_class_iu,
    save_checkpoint,
    load_checkpoint
)

import torch.nn as nn
import os
from matplotlib import pyplot as plt
EPOCHS = 100
LEARNING_RATE = 0.01
#DEVICE = "cuda:0,1" #if torch.cuda.is_available() else "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
BATCH_SIZE = 1
CHECKPOINT_STEP = 2
VALIDATE_STEP = 1
NUM_CLASSES = 3
LOAD_MODEL =  False

#############------Dataloader-----##################


#-------------Dataloader-------------#
dataset = dataloader.Image_loader()
total_samples = len(dataset)
print(total_samples)
train_split = int(0.8 * total_samples)

train_set, val_set = torch.utils.data.random_split(dataset, [train_split, len(dataset)-train_split])

dataloader_train = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers = 4,
    pin_memory = True
)
dataloader_val = DataLoader(
    val_set,
    batch_size=1,
    shuffle=True,
    num_workers = 4,
)





###############-----ADD model-----##################
model = DeepLabV3()
model = model.to(device=DEVICE)

###############---Optimizer---##################
optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
# optimizer = torch.optim.Adam(train_params, momentum=0.9,
#                             weight_decay=5e-4, nesterov=False)

##########---Loss Function---##############
loss_func = torch.nn.CrossEntropyLoss()

#Validate (Đánh giá)
def val(model, dataloader):
    accuracy_arr = []
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
    with torch.no_grad():
        model.eval()
        print('validating...')

        for i, (val_data, val_label) in enumerate(dataloader):
            val_data = val_data.to(device=DEVICE)
            #output of the model is (1,num_classes,W,H) => (num_classes,W,H)
            val_output = model(val_data).squeeze()

            #convert (nc,W,H) => (W,H) with one hot encoder

            val_output = reverse_one_hot(val_output)
            val_output = np.array(val_output.cpu())
            #Process label and convert to (W,H) imaget
            val_label = val_label.squeeze()
            val_label = np.array(val_label.cpu())
            #Compute acc and iou
            accuracy = compute_accuracy(val_output, val_label)
            # print('acc?',accuracy)
            hist += fast_hist(val_label.flatten(),val_output.flatten(), NUM_CLASSES)
            #Append to calculate
            accuracy_arr.append(accuracy)

        miou_list = per_class_iu(hist)[:-1]
        mean_accuracy, mean_iou = np.mean(accuracy_arr), np.mean(miou_list)
        print('Mean_accuracy:{} mIOU:{}'.format(mean_accuracy, mean_iou))
        return mean_accuracy, mean_iou

#Training
torch.cuda.empty_cache()

max_iou = 0
trainning_accuracy_segment=[]
trainning_loss=[]
for epoch in range(EPOCHS):
    
    model.train()
    tq = tqdm(total=len(dataloader_train) * BATCH_SIZE)
    tq.set_description('Epoch {}/{}'.format(epoch, EPOCHS))

    loss_record = []
    loss=0

    for i, (data, label) in enumerate(dataloader_train):
        data = data.to(device=DEVICE)
        label = label.to(device=DEVICE)
        output = model(data)
        #print(output.size())
        #print(label.size())
        loss = loss_func(output, label)
        #print(loss)
        #Combine 3 losses
        tq.update(BATCH_SIZE)
        tq.set_postfix(loss='%6f'%loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_record.append(loss.item())
    tq.close()
    loss_train_mean = np.mean(loss_record)
    print('loss for train: %f'%(loss_train_mean))
    trainning_loss.append(loss_train_mean)
    plt.figure(1)
    plt.title("Training mIoU segment Graph")
    plt.grid()
    plt.plot(trainning_accuracy_segment, color = 'g') # plotting training loss
    plt.legend(['mIoU'], loc='upper left')
    plt.savefig('plot_mious_segment.png')
    plt.grid()
    ##########################################################
    plt.figure(2)
    plt.title("Training loss Graph")
    plt.grid()
    plt.plot(trainning_loss, color = 'b') # plotting training loss
    plt.legend(['loss'], loc='upper right')
    
    plt.savefig('plot_loss.png')
    plt.grid()

    #save checkpoint
    # if epoch % CHECKPOINT_STEP == 0:
    #     torch.save(model.state_dict(), 'checkpoints/best_model.pth')

    #validate and save ckp
    if epoch % VALIDATE_STEP == 0:
        _, mean_iou = val(model, dataloader_val)
        trainning_accuracy_segment.append(mean_iou)
        
        if mean_iou > max_iou:
            max_iou = mean_iou
            print('save best model with mIOU = {}'.format(mean_iou))
            torch.save(model.state_dict(), 'Trained_Model/test1_model.pth')
            # checkpoint = {
            #     "state_dict": model.state_dict(),
            #     "optimizer":optimizer.state_dict(),
            # }
            # save_checkpoint(checkpoint, 'checkpoints/best_model_01.pth'
