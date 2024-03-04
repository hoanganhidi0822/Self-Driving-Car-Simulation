# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from model.backbone import resnet
from model.ASPP_new import ASPP
from model.decoder import Decoder
# from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
# from backbone import resnet
# from ASPP_new import ASPP
# from decoder import Decoder
from torchsummary import summary
class DeepLabV3(nn.Module):
    def __init__(self):
        super(DeepLabV3, self).__init__()

        self.num_classes = 4



        BatchNorm = nn.BatchNorm2d
        self.resnet = resnet(layers = '18',pretrained = True)
        self.aspp = ASPP(BatchNorm= BatchNorm)
        
        self.decoder = Decoder(self.num_classes, BatchNorm = BatchNorm)
    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))
        h = x.size()[2]
        w = x.size()[3]

        x2,x3,fea, low_feature = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16))
        fea_aspp = self.aspp(fea) # (shape: (batch_size, num_classes, h/16, w/16))
        #print(fea_aspp.size())
        #output = self.decoder(fea_aspp,low_feature)
        output = F.interpolate(fea_aspp, size=(h, w), mode='bilinear', align_corners=True)
        #print('output',output.size())

        #output = F.interpolate(fea, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))
        #print(output.size())

        return output
# model_deeplab = DeepLabV3().cuda()
# print(summary(model_deeplab, (3,288,800), 2))

