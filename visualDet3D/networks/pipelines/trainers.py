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
from visualDet3D.utils.utils import LossLogger
from visualDet3D.utils.utils import compound_annotation
from visualDet3D.networks.utils.registry import PIPELINE_DICT


@PIPELINE_DICT.register_module
def train_mono_detection(data, module:nn.Module,
                         optimizer:optim.Optimizer,
                         writer:SummaryWriter=None,
                         loss_logger:LossLogger=None,
                         global_step:int=None,
                         epoch_num:int=None,
                         cfg:EasyDict=EasyDict()):
    optimizer.zero_grad()
    # load data
    image, road_maps, calibs, labels, bbox2d, bbox_3d = data

    # create compound array of annotation
    max_length = np.max([len(label) for label in labels])
    if max_length == 0:
        return
    annotation = compound_annotation(labels, max_length, bbox2d, bbox_3d, cfg.obj_types) #np.arraym, [batch, max_length, 4 + 1 + 7]

    # Feed to the network
    classification_loss, regression_loss, loss_dict, cls_preds, reg_preds, anchors = module(
            [image.cuda().contiguous(), road_maps.cuda(), image.new(annotation).cuda(), calibs.cuda()])

    classification_loss = classification_loss.mean()
    regression_loss = regression_loss.mean()

    # Record loss in a average meter
    if loss_logger is not None:
        loss_logger.update(loss_dict)
        # assert batch_size=1 is only applied for cls_preds. Doing anything
        # else just for consistency.

        if not global_step % 100:
            bounding_boxes = module.get_bboxes(cls_preds[:1], reg_preds[:1], anchors,
                                               calibs[:1].cuda(), image[:1].cuda().contiguous())
            loss_logger.log_image(image, bounding_boxes, road_maps, annotation, calibs, global_step)

    loss = classification_loss + regression_loss

    if bool(loss.item() == 0):
        return
    loss.backward()
    # clip loss norm
    torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.optimizer.clipped_gradient_norm)

    optimizer.step()
    optimizer.zero_grad()
