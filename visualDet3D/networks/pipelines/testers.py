"""
    This script contains function snippets for different training settings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from easydict import EasyDict
from visualDet3D.utils.utils import LossLogger, compound_annotation
from visualDet3D.networks.utils.registry import PIPELINE_DICT
from typing import Tuple, List

@PIPELINE_DICT.register_module
@torch.no_grad()
def test_mono_detection(data, module:nn.Module,
                     writer:SummaryWriter,
                     loss_logger:LossLogger=None,
                     global_step:int=None,
                     cfg:EasyDict=None)-> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    image, road_maps, P2 = data[0], data[1], data[2]
    labels, bbox2d, bbox_3d = data[3], data[4], data[5]

    scores, bbox, obj_index = module(
        [image.cuda().float().contiguous(), road_maps.cuda(), P2.clone().cuda().float()])
    obj_types = [cfg.obj_types[i.item()] for i in obj_index]
    if loss_logger is not None:
        max_length = np.max([len(label) for label in labels])
        annotation = compound_annotation(labels, max_length, bbox2d, bbox_3d, cfg.obj_types)
        loss_logger.log_image(image, bbox, road_maps, annotation, P2, global_step)

    return scores, bbox, obj_types
