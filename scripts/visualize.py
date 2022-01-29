import cv2
import numpy as np
import torch

from ptools import vis
from visualDet3D.data.kitti.dataset.mono_dataset import KittiMonoDataset
from visualDet3D.data.synthia.mono_dataset import SynthiaMonoDataset
from visualDet3D.utils.utils import draw_3D_box
from visualDet3D.utils.utils import cfg_from_file
from visualDet3D.utils.visualize_utils import draw_bbox2d_to_image

from visualDet3D.networks.utils import BBox3dProjector, BackProjection

synthia_cfg = cfg_from_file('config/Road_synthia_train.py')
kitti_cfg = cfg_from_file('config/Road_kitti_train.py')
synthia_ds = SynthiaMonoDataset(synthia_cfg)
kitti_ds = KittiMonoDataset(kitti_cfg)

projector = BBox3dProjector()
backprojector = BackProjection()


def denorm(image, cfg):
    new_image = np.array(
        (image * cfg.data.augmentation.rgb_std +  cfg.data.augmentation.rgb_mean) * 255, dtype=np.uint8)
    return new_image


def draw_center(image, bboxs, color):
    for bbox in bboxs:
        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]).astype(np.int)
        image = cv2.circle(image, center, 1, color, thickness=5)
    return image


for idx, sample in enumerate(synthia_ds):
    if idx < 100:
        continue
    image = sample['image']
    P2 = torch.Tensor(sample['calib'])
    bbox2d = sample['bbox2d']
    bbox3d = sample['bbox3d']
    bbox_3d_state_3d = backprojector(torch.Tensor(bbox3d), P2)
    abs_bbox, bbox_3d_corner_homo, thetas = projector(bbox_3d_state_3d, P2)
    print(bbox_3d_corner_homo.shape)
    # import pdb
    # pdb.set_trace()
    image_copy = denorm(image, synthia_cfg)

    for box in bbox_3d_corner_homo:
        image_copy = draw_3D_box(image_copy, box.numpy().transpose(1, 0), color=(0, 0, 255))
    image_copy = draw_bbox2d_to_image(image_copy, bbox2d, color=(255, 0, 0), thickness=2)
    image_copy = draw_center(image_copy, bbox2d, color=(255, 0, 0))
    for bbox in bbox3d[:, :2]:
        image_copy = cv2.circle(image_copy, bbox.astype(np.int), 1, color=(0, 0, 255), thickness=5)

    vis.np3ds(image_copy)
    input()
