import numpy as np
import torch.nn as nn
import torch
import math
import time
from visualDet3D.networks.backbones import resnet


class YoloMono3DCore(nn.Module):
    """Some Information about YoloMono3DCore"""
    def __init__(self, backbone_arguments=dict()):
        super(YoloMono3DCore, self).__init__()
        self.backbone =resnet(**backbone_arguments)

    def forward(self, x):
        x = self.backbone(x['image'])
        x = x[0]
        return x


class YoloRoadMono3DCore(nn.Module):
    """Some Information about YoloMono3DCore"""
    def __init__(self, backbone_arguments=dict()):
        super(YoloRoadMono3DCore, self).__init__()
        self.backbone = resnet(**backbone_arguments)

    def forward(self, x):
        batch_size = x['image'].shape[0]
        image = x['image']
        road_maps = x['road_maps'].unsqueeze(1).repeat(1, 3, 1, 1) / torch.max(x['road_maps'])
        image_features = self.backbone(image)[0]
        road_features = self.backbone(road_maps)[0]
        # import pdb
        # pdb.set_trace()
        features = torch.cat([image_features, road_features], dim=1)
        return features
