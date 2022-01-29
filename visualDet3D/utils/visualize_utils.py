import cv2
import numpy as np
import sys
import torch
from torch.utils.data import DataLoader


sys.path.append("../")

from visualDet3D.networks.utils import BackProjection, BBox3dProjector
from visualDet3D.data.kitti.dataset import mono_dataset
from visualDet3D.utils.utils import cfg_from_file
from visualDet3D.utils.utils import compound_annotation

backprojector = BackProjection()
projector = BBox3dProjector()


def draw_bbox2d_to_image(image, bboxes2d, color=(0, 0, 0), thickness=3):
    drawed_image = image.copy()
    for box2d in bboxes2d:
        drawed_image = cv2.rectangle(
            drawed_image, (int(box2d[0]), int(box2d[1])), (int(box2d[2]), int(box2d[3])), color, thickness)
    return drawed_image


def draw_points(image, bottom_points, color=(0, 0, 255), thickness=3):
    drawed_image = image.copy()
    for bottom_point in bottom_points:
        drawed_image = cv2.circle(
            drawed_image, (int(bottom_point[0]), int(bottom_point[1])),
            radius=1, color=color, thickness=thickness)
    return drawed_image


def anchor_to_bbox(anchors):
    """
    params:
        anchors = np.ndarray(N, 4)  # (x, y, w, l)
    return:
        bboxes = np.ndarray(N, 4)  # (x1, y1, x2, y2)
    """
    bboxes = np.ones(anchors.shape)
    bboxes[:, 0] = anchors[:, 0] - anchors[:, 2] / 2
    bboxes[:, 1] = anchors[:, 1] - anchors[:, 3] / 2
    bboxes[:, 2] = anchors[:, 0] + anchors[:, 2] / 2
    bboxes[:, 3] = anchors[:, 1] + anchors[:, 3] / 2
    return bboxes


def get_bev_bboxes(annotations, P2):
    if annotations.shape[0] == 0:
        return np.ones((0, 2, 2))
    if annotations.shape[1] == 11:
        bbox_3d_state = annotations[:, 4:]
    else:
        bbox_3d_state = annotations[:, 5:]
    bbox_3d_state = bbox_3d_state[torch.any(bbox_3d_state != -1, 1)]
    bbox_3d_state_3d = backprojector(bbox_3d_state, P2)
    abs_bbox, bbox_3d_corner_homo, thetas = projector(bbox_3d_state_3d, P2)

    # mean_x = bbox_3d_state_3d[:, 0].mean()
    # mean_z = bbox_3d_state_3d[:, 2].mean()

    scale = 10
    bbox_3d_state_3d[:, 0] = (bbox_3d_state_3d[:, 0]) * scale
    bbox_3d_state_3d[:, 2] = (bbox_3d_state_3d[:, 2]) * scale
    scale = 5
    bbox_3d_state_3d[:, 3:] = bbox_3d_state_3d[:, 3:] * scale
    points = []
    for bbox_3d in bbox_3d_state_3d:
        x, y, z, w, h, l, alpha = bbox_3d.numpy()
        point = np.array([[x - w/2, z - l/2], [x + w/2, z + l/2]])
        points.append(point)
    if not points:
        return np.ones((0, 2, 2))
    points = np.stack(points).astype(np.int)
    return points


def add_bboxes_to_image(bboxes, shape=(800, 800, 3), color=(0, 0, 0), image=None):
    if type(color) == tuple:
        color = [color] * len(bboxes)
    for idx, points in enumerate(bboxes):
        image = np.ones(shape).astype(np.uint8) * 255 if image is None else image
        mean_x = np.abs(points[:, :, 0].mean()).astype(np.int)
        mean_z = np.abs(points[:, :, 1].mean()).astype(np.int)
        points[:, :, 0] += int(shape[0] / 2) - mean_x
        points[:, :, 1] += int(shape[1] / 2) - mean_z
        for bbox in points:
            image = cv2.rectangle(image, bbox[0], bbox[1], color[idx], 1)
    return image


def get_bev_pred_gt(pred, gt, P2, shape=(800, 800, 3), colors=[(255, 0, 0), (0, 0, 255)]):
    bboxes = [get_bev_bboxes(pred, P2), get_bev_bboxes(gt, P2)]
    return add_bboxes_to_image(bboxes, color=colors)


def get_bev_image(annotations, P2, shape=(800, 800, 3), color=(0, 0, 0), image=None):
    image = np.ones(shape).astype(np.uint8) * 255 if image is None else image
    if annotations.shape[0] == 0:
        return image
    if annotations.shape[1] == 11:
        bbox_3d_state = annotations[:, 4:]
    else:
        bbox_3d_state = annotations[:, 5:]
    bbox_3d_state = bbox_3d_state[torch.any(bbox_3d_state != -1, 1)]
    bbox_3d_state_3d = backprojector(bbox_3d_state, P2)
    abs_bbox, bbox_3d_corner_homo, thetas = projector(bbox_3d_state_3d, P2)

    mean_x = bbox_3d_state_3d[:, 0].mean()
    mean_z = bbox_3d_state_3d[:, 2].mean()

    scale = 10
    bbox_3d_state_3d[:, 0] = (bbox_3d_state_3d[:, 0] - mean_x) * scale
    bbox_3d_state_3d[:, 2] = (bbox_3d_state_3d[:, 2] - mean_z) * scale
    scale = 5
    bbox_3d_state_3d[:, 3:] = bbox_3d_state_3d[:, 3:] * scale
    points = []
    for bbox_3d in bbox_3d_state_3d:
        x, y, z, w, h, l, alpha = bbox_3d.numpy()
        point = np.array([[x - w/2, z - l/2], [x + w/2, z + l/2]])
        points.append(point)
    if not points:
        return image
    points = np.stack(points).astype(np.int)
    mean_x = np.abs(points[:, :, 0].mean()).astype(np.int)
    mean_z = np.abs(points[:, :, 1].mean()).astype(np.int)
    points[:, :, 0] += int(shape[0] / 2) - mean_x
    points[:, :, 1] += int(shape[1] / 2) - mean_z
    for bbox in points:
        image = cv2.rectangle(image, bbox[0], bbox[1], color, 1)
    return image


def get_corners(bbox):
    # w, h, l, y, z, x, yaw = bbox
    y, z, x, w, h, l, yaw = bbox
    y = -y
    # manually take a negative s. t. it's a right-hand
    # system, with
    # x facing in the front windshield of the
    # car
    # z facing up
    # y facing to the left of
    # driver

    # yaw = -(yaw + np.pi / 2)
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    # rear
    # left
    bev_corners[0, 0] = x - l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
    bev_corners[0, 1] = y - l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

    # rear
    # right
    bev_corners[1, 0] = x - l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
    bev_corners[1, 1] = y - l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

    # front
    # right
    bev_corners[2, 0] = x + l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
    bev_corners[2, 1] = y + l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

    # front
    # left
    bev_corners[3, 0] = x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
    bev_corners[3, 1] = y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

    return bev_corners


if __name__ == '__main__':
    cfg = cfg_from_file("/home/shanus/workspace/3D_detection/visualDet3D/config/Road_kitti_debug.py")
    dataset = mono_dataset.KittiMonoDataset(cfg)

    dataloader_train = DataLoader(dataset, num_workers=cfg.data.num_workers,
                                  batch_size=cfg.data.batch_size, collate_fn=dataset.collate_fn)

    idx = 5
    for index, data in enumerate(dataloader_train):
        image, road_maps, calibs, labels, bbox2d, bbox_3d = data
        max_length = np.max([len(label) for label in labels])
        annotation = compound_annotation(labels, max_length, bbox2d, bbox_3d, cfg.obj_types)
        bev_image = get_bev_image(torch.Tensor(annotation[idx]), torch.Tensor(calibs[idx]))
        cv2.imwrite('bev' + str(index) + '.png', bev_image)
        cv2.imwrite('image' + str(index) + '.png', image[idx].numpy().transpose(1, 2, 0))
        if index > 8:
            break
