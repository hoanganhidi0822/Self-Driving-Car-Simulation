# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import os

from model.backbone import resnet
from model.ASPP import ASPP
#from model.ASPP_new import ASPP
# from model.decoder import build_decoder
# from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
class DeepLabV3(nn.Module):
    def __init__(self,sync_bn=True):
        super(DeepLabV3, self).__init__()

        self.num_classes = 3

        BatchNorm = nn.BatchNorm2d
        self.resnet = resnet(layers = '18',pretrained = True) 
        self.aspp = ASPP(num_classes= self.num_classes)# (batchsize,256,9,25)
        #self.aspp = ASPP(BatchNorm= BatchNorm)
        
        #self.decoder = build_decoder(self.num_classes, backbone ='resnet', BatchNorm = BatchNorm)
    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        x2,x3,fea, low_feature = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        fea = self.aspp(fea) # (shape: (batch_size, num_classes, h/16, w/16))

        output = F.interpolate(fea, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))
        #print(output.size())

        return output

#model = DeepLabV3().cuda()
#print(summary(model, (3,80,160),1))