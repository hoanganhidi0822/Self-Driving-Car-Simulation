import torch,pdb
import torchvision
import torch.nn.modules


class resnet(torch.nn.Module):
    def __init__(self,layers,pretrained = False):
        super(resnet,self).__init__()
        if layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == '50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif layers == '101':
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif layers == '152':
            model = torchvision.models.resnet152(pretrained=pretrained)
        elif layers == '50next':
            model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        elif layers == '101next':
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        elif layers == '50wide':
            model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        elif layers == '101wide':
            model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self,x):
        #print('original',x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print('layer1',x.size())
        x = self.maxpool(x)
        #print('after maxpool',x.size())
        x = self.layer1(x)
        low_level_feat = x
        #print('low level', low_level_feat.size())
        #print('after layer 1',x.size())
        x2 = self.layer2(x)
        #print('after layer 2',x2.size())
        x3 = self.layer3(x2)
        #print('after layer 3',x3.size())
        x4 = self.layer4(x3)
        #print('after layer 4',x4.size())
        return x2,x3,x4, low_level_feat
#model = resnet(layers = '18',pretrained = True).cuda()
#print(summary(model,(3, 80, 160)))