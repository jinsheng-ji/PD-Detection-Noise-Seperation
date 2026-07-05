# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights,ResNet50_Weights

class resnet50(nn.Module):

    def __init__(self, pretrained=True, num_classes=3):
        super(resnet50, self).__init__()
        features_tmp = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        features = torch.nn.Sequential(*list(features_tmp.children())[:-2])
        
        self.features = features
        self.fc_cl = nn.Linear(512*4, num_classes)
        self.fc_ce = nn.Linear(512*4, 2)
        self.averpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b,c,w,h = x.size()
        x = self.features(x)
        x_fea = self.averpool(x).view(b, -1)
        x_out = self.fc_cl(x_fea)
        x_fea_ce = self.fc_ce(x_fea)

        return x_out,x_fea,x_fea_ce
