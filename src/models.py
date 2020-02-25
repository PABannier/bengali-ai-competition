import torch
import torch.nn as nn
from torch.nn import functional as F
import pretrainedmodels
from efficientnet_pytorch import EfficientNet
import numpy as np

# Implement the Freeze and unfreeze functions

class ResNet34(nn.Module):
    def __init__(self, pretrained):
        super(ResNet34, self).__init__()

        if pretrained:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)
        
        self.l0 = nn.Linear(512, 168)
        self.l1 = nn.Linear(512, 11)
        self.l2 = nn.Linear(512, 7)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)
        
        return l0, l1, l2


class SE_ResNeXt50_32x4d(nn.Module):
    def __init__(self, pretrained):
        super(SE_ResNeXt50_32x4d, self).__init__()

        if pretrained:
            self.model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet')

        out_dim = self.model.last_linear.in_features
        
        self.l0 = nn.Linear(out_dim, 168)
        self.l1 = nn.Linear(out_dim, 11)
        self.l2 = nn.Linear(out_dim, 7)
    
    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)

        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)

        return l0, l1, l2


class EfficientNetB3(nn.Module):
    def __init__(self, pretrained):
        super(EfficientNetB3, self).__init__()

        if pretrained:
            self.model = EfficientNet.from_pretrained("efficientnet-b3")
        else:
            self.model = EfficientNet.from_name("efficientnet-b3")
        

        self.l0 = nn.Linear(1000, 168)
        self.l1 = nn.Linear(1000, 11)
        self.l2 = nn.Linear(1000, 7)

    def forward(self, x):
        x = self.model(x)

        l0 = self.l0(x)
        l1 = self.l1(x)
        l2 = self.l2(x)

        return l0, l1, l2