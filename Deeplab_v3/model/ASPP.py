import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

using_cbam = True

class ASPP(nn.Module):
    def __init__(self, num_classes, channels=512):  # Set channels to 256
        super(ASPP, self).__init__()

        reduced_channels = channels // 32

        self.conv_1x1_1 = nn.Conv2d(512, reduced_channels, kernel_size=1)  # Correct the input channels to 256
        self.bn_conv_1x1_1 = nn.BatchNorm2d(reduced_channels)

        self.conv_3x3_1 = nn.Conv2d(channels, reduced_channels, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(reduced_channels)

        self.conv_3x3_2 = nn.Conv2d(channels, reduced_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(reduced_channels)

        self.conv_3x3_3 = nn.Conv2d(channels, reduced_channels, kernel_size=3, stride=1, padding=4, dilation=4)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(reduced_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(reduced_channels)

        self.conv_1x1_3 = nn.Conv2d(reduced_channels * 5, reduced_channels, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(reduced_channels)

        self.conv_1x1_4 = nn.Conv2d(reduced_channels, num_classes, kernel_size=1)
        self._init_weight()

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        out_img = self.avg_pool(feature_map)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear")

        #print(out_1x1.size())

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)

        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))
        out = self.conv_1x1_4(out)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

