import torch
import torch.nn as nn
import torch.nn.functional as F
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from model.ASPP import build_aspp
from model.decoder import build_decoder
from model.backbone import resnet

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=8,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        self.num_classes = 4
        self.resnet = resnet(layers = '18',pretrained = True)

        self.aspp = build_aspp(backbone ='resnet', output_stride = 8, BatchNorm = BatchNorm)
        self.decoder = build_decoder(self.num_classes, backbone ='resnet', BatchNorm = BatchNorm)
        


        # input : nchw,
        # output: (w+1) * sample_rows * 4 
        #self.model = resnet(backbone, pretrained=pretrained)
       

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x2,x3,x, low_level_feat = self.resnet(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()
    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p






