from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import os

from visualDet3D.utils.utils import alpha2theta_3d
from visualDet3D.networks.heads.anchors import Anchors


def ravel_multi_index(bottom_points, road_maps):
    mf = torch.tensor([[1], [road_maps.shape[2]]]).cuda()
    return (bottom_points.int() * mf).sum(axis=0)


class RoadBasedAnchors(Anchors):
    """ Anchor modules for multi-level dense output.

    """

    def __init__(self, **kwargs):
        super(RoadBasedAnchors, self).__init__(**kwargs)
        self.use_z = kwargs['use_z']
        self.location = kwargs['location']

    def get_anchors_z_mixed_3_sided(self, road_maps):

        if not self.use_z:
            return self.anchor_mean_std[:, :, :, 0, 0].permute(0, 2, 1)
        B, H, W = road_maps.shape
        # anchors.shape = class_type, num_anchors, params
        image_x_left_corner = self.anchors[0, :, 2]
        image_x_right_corner = self.anchors[0, :, 0]

        left_side_points = torch.stack([image_x_left_corner, self.anchors_image_y_center]).cuda()
        left_side_points[0, :] = torch.clamp(left_side_points[0, :], min=0, max=W - 1)
        left_side_points[1, :] = torch.clamp(left_side_points[1, :], min=0, max=H - 1)
        left_side_points_ravelled = ravel_multi_index(left_side_points, road_maps)
        anchors_z = torch.stack(
            [torch.take(road_maps[idx], left_side_points_ravelled) for idx in range(road_maps.shape[0])])

        right_side_points = torch.stack([image_x_right_corner, self.anchors_image_y_center]).cuda()
        right_side_points[0, :] = torch.clamp(right_side_points[0, :], min=0, max=W - 1)
        right_side_points[1, :] = torch.clamp(right_side_points[1, :], min=0, max=H - 1)
        right_side_points_ravelled = ravel_multi_index(right_side_points, road_maps)
        anchors_z += torch.stack(
            [torch.take(road_maps[idx], right_side_points_ravelled) for idx in range(road_maps.shape[0])])

        bottom_points = torch.stack([self.anchors_image_x_center, self.anchors[0, :, 3]]).cuda()
        bottom_points[1, :] = torch.clamp(bottom_points[1, :], min=0, max=H - 1)
        bottom_points[0, :] = torch.clamp(bottom_points[0, :], min=0, max=W - 1)
        bottom_points_ravelled = ravel_multi_index(bottom_points, road_maps)
        road_maps = road_maps + (self.anchor_mean_std[:, :, 0, 5, 0] / 2).mean()
        anchors_z = torch.stack(
            [torch.take(road_maps[idx], bottom_points_ravelled) for idx in range(road_maps.shape[0])])

        anchors_z = anchors_z / 3

        mask = road_maps == 0 # True means non-road, False means road
        mask = torch.stack(
            [torch.take(mask[idx], bottom_points_ravelled) for idx in range(mask.shape[0])])
        anchors_z[mask] = 0
        anchors_z_2 = self.anchor_mean_std[:, :, :, 0, 0].permute(0, 2, 1).squeeze(1)
        anchors_z_2[~mask] = 0
        anchors_z += anchors_z_2
        return anchors_z.unsqueeze(1)

    def get_anchors_z_mixed(self, road_maps):

        if not self.use_z:
            return self.anchor_mean_std[:, :, :, 0, 0].permute(0, 2, 1)
        B, H, W = road_maps.shape

        mask = road_maps == 0 # True means non-road, False means road

        bottom_points = torch.stack([self.anchors_image_x_center, self.anchors[0, :, 3]]).cuda()
        bottom_points[1, :] = torch.clamp(bottom_points[1, :], min=0, max=287)

        road_maps = road_maps + (self.anchor_mean_std[:, :, 0, 5, 0] / 2).mean()
        bottom_points_ravelled = ravel_multi_index(bottom_points, road_maps)
        anchors_z = torch.stack(
            [torch.take(road_maps[idx], bottom_points_ravelled) for idx in range(road_maps.shape[0])])
        mask = torch.stack(
            [torch.take(mask[idx], bottom_points_ravelled) for idx in range(mask.shape[0])])
        anchors_z[mask] = 0
        anchors_z_2 = self.anchor_mean_std[:, :, :, 0, 0].permute(0, 2, 1).squeeze(1)
        anchors_z_2[~mask] = 0
        anchors_z += anchors_z_2
        return anchors_z.unsqueeze(1)

    def get_anchors_z_simple_grid_sample(self, road_maps):
        if not self.use_z:
            return self.anchor_mean_std[:, :, :, 0, 0].permute(0, 2, 1)
        B, H, W = road_maps.shape
        bottom_points = torch.stack([self.anchors[0, :, 3], self.anchors_image_x_center]).cuda()
        # bottom_points[1, :] = torch.clamp(bottom_points[1, :], min=0, max=287)

        # putting the z at the center.
        # This is a hacky way to do it. Just adding vehicle's length to the
        # entire depth map.
        # self.anchor_mean_std = (B, N, types, (z, sina, cosa, w, h, l), (mean, std))
        road_maps = road_maps + (self.anchor_mean_std[:, :, 0, 5, 0] / 2).mean() # B, H, W

        bottom_points = bottom_points.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1).permute(0, 1, 3, 2) # B, 1, N, 2
        bottom_points[:, :, :, 0] = (bottom_points[:, :, :, 0] - H/2 - 1) / (H/2)
        bottom_points[:, :, :, 1] = (bottom_points[:, :, :, 1] - W/2 - 1) / (W/2)
        road_maps = road_maps.unsqueeze(1) # B, 1, H, W

        anchors_z = torch.nn.functional.grid_sample(road_maps, bottom_points, padding_mode='reflection') # B, 1, 1, N
        return anchors_z.squeeze(1)

    def get_anchors_z_from_3_sides_grid_sample(self, road_maps):
        if not self.use_z:
            return self.anchor_mean_std[:, :, :, 0, 0].permute(0, 2, 1)
        B, H, W = road_maps.shape
        road_maps = road_maps.unsqueeze(1)

        # anchors.shape = class_type, num_anchors, params
        image_x_left_corner = self.anchors[0, :, 2]
        image_x_right_corner = self.anchors[0, :, 0]
        image_x_left_corner = (image_x_left_corner.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1) - W/2 - 1) / (W/2)
        image_x_right_corner = (image_x_right_corner.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1) - W/2 - 1) / (W/2)

        anchors_image_x_center = (
            self.anchors_image_x_center.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1) - W/2 - 1) / (W/2)
        anchors_image_y_center = (
            self.anchors_image_y_center.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1) - H/2 - 1) / (H/2)

        left_side_points = torch.stack([anchors_image_y_center, image_x_left_corner]).cuda().permute(1, 2, 3, 0)
        anchors_z = torch.nn.functional.grid_sample(road_maps, left_side_points, padding_mode='reflection') # B, 1, 1, N

        right_side_points = torch.stack([anchors_image_y_center, image_x_right_corner]).cuda().permute(1, 2, 3, 0)
        anchors_z += torch.nn.functional.grid_sample(
            road_maps, right_side_points, padding_mode='reflection') # B, 1, 1, N

        bottom_points = torch.stack([anchors_image_y_center, anchors_image_x_center]).cuda().permute(1, 2, 3, 0)
        anchors_z += torch.nn.functional.grid_sample(road_maps, bottom_points, padding_mode='reflection') # B, 1, 1, N
        return anchors_z.squeeze(1) / 3

    def get_anchors_z(self, road_maps):
        if not self.use_z:
            return self.anchor_mean_std[:, :, :, 0, 0].permute(0, 2, 1)
        B, H, W = road_maps.shape
        image_x_corner = self.anchors[0, :, 0]

        bottom_points = torch.stack([image_x_corner, self.anchors[0, :, 3]]).cuda()
        bottom_points[1, :] = torch.clamp(bottom_points[1, :], min=0, max=287)
        bottom_points_ravelled = ravel_multi_index(bottom_points, road_maps)

        anchors_z = torch.stack(
            [torch.take(road_maps[idx], bottom_points_ravelled) for idx in range(road_maps.shape[0])])
        anchors_z += calculate_z_offset(
            image_x_corner, anchors_z, self.P2, self.anchor_mean_std[0, :, 0, :, 0])
        return anchors_z.unsqueeze(1)

    def get_anchors_z_simple(self, road_maps):
        if not self.use_z:
            return self.anchor_mean_std[:, :, :, 0, 0].permute(0, 2, 1)
        B, H, W = road_maps.shape
        bottom_points = torch.stack([self.anchors_image_x_center, self.anchors[0, :, 3]]).cuda()
        bottom_points[1, :] = torch.clamp(bottom_points[1, :], min=0, max=287)

        # putting the z at the center.
        # This is a hacky way to do it. Just adding vehicle's length to the
        # entire depth map.
        # self.anchor_mean_std = (B, N, types, (z, sina, cosa, w, h, l), (mean, std))
        road_maps = road_maps + (self.anchor_mean_std[:, :, 0, 5, 0] / 2).mean()

        bottom_points_ravelled = ravel_multi_index(bottom_points, road_maps)
        anchors_z = torch.stack(
            [torch.take(road_maps[idx], bottom_points_ravelled) for idx in range(road_maps.shape[0])])
        anchors_z = anchors_z.unsqueeze(1)
        return anchors_z

    def get_anchors_z_from_3_sides(self, road_maps):
        if not self.use_z:
            return self.anchor_mean_std[:, :, :, 0, 0].permute(0, 2, 1)
        B, H, W = road_maps.shape
        # anchors.shape = class_type, num_anchors, params
        image_x_left_corner = self.anchors[0, :, 2]
        image_x_right_corner = self.anchors[0, :, 0]

        left_side_points = torch.stack([image_x_left_corner, self.anchors_image_y_center]).cuda()
        left_side_points[0, :] = torch.clamp(left_side_points[0, :], min=0, max=W - 1)
        left_side_points[1, :] = torch.clamp(left_side_points[1, :], min=0, max=H - 1)
        left_side_points_ravelled = ravel_multi_index(left_side_points, road_maps)
        anchors_z = torch.stack(
            [torch.take(road_maps[idx], left_side_points_ravelled) for idx in range(road_maps.shape[0])])

        right_side_points = torch.stack([image_x_right_corner, self.anchors_image_y_center]).cuda()
        right_side_points[0, :] = torch.clamp(right_side_points[0, :], min=0, max=W - 1)
        right_side_points[1, :] = torch.clamp(right_side_points[1, :], min=0, max=H - 1)
        right_side_points_ravelled = ravel_multi_index(right_side_points, road_maps)
        anchors_z += torch.stack(
            [torch.take(road_maps[idx], right_side_points_ravelled) for idx in range(road_maps.shape[0])])

        bottom_points = torch.stack([self.anchors_image_x_center, self.anchors[0, :, 3]]).cuda()
        bottom_points[1, :] = torch.clamp(bottom_points[1, :], min=0, max=H - 1)
        bottom_points[0, :] = torch.clamp(bottom_points[0, :], min=0, max=W - 1)
        bottom_points_ravelled = ravel_multi_index(bottom_points, road_maps)
        road_maps = road_maps + (self.anchor_mean_std[:, :, 0, 5, 0] / 2).mean()
        anchors_z = torch.stack(
            [torch.take(road_maps[idx], bottom_points_ravelled) for idx in range(road_maps.shape[0])])

        anchors_z = anchors_z / 3

        return anchors_z.unsqueeze(1)


def visualize_anchors(anchor_obj, image, road_maps):
    import random
    from visualDet3D.utils import visualize_utils
    blank_image = np.ones((1600, 1600, 3)) * 255
    selected_indices = random.sample(range(anchor_obj.anchors.shape[1]), 100)
    # selected_indices = [18978]
    anchors1 = anchor_obj.anchors.cpu().numpy()[0, selected_indices, :]
    anchors1[:, :2] = anchors1[:, :2]
    # selected_bboxes = visualize_utils.anchor_to_bbox(anchors1)
    selected_bboxes = anchors1

    selected_bottom_points = np.stack(
        [anchor_obj.anchors_image_x_center.cpu().numpy(),
         anchor_obj.anchors[0, :, 3].cpu().numpy()])[:, selected_indices]

    selected_bottom_points = selected_bottom_points.transpose(1, 0)
    # selected_bottom_points = selected_bottom_points + 800
    drawed_image = visualize_utils.draw_bbox2d_to_image(blank_image, selected_bboxes)
    drawed_image = visualize_utils.draw_points(drawed_image, selected_bottom_points, thickness=10)
    # cv2.imwrite('drawed_image.png', drawed_image)
    return drawed_image


def calculate_z_offset(image_x_corner, z3d, P2, anchor_means):
    cx = P2[0, 0:1, 2:3].squeeze(-1)
    fy = P2[0, 1:2, 1:2].squeeze(-1)
    alpha = torch.atan2(anchor_means[..., 1], anchor_means[..., 2]) / 2.0
    x3d = (image_x_corner - z3d.new(cx)) * z3d / z3d.new(fy) #[B, types, N]
    # x3d = (image_x_corner - cx) * z3d / fy
    theta = alpha2theta_3d(alpha, x3d, z3d, P2[0])
    z_offset = anchor_means[..., 3] * torch.cos(theta) + anchor_means[..., 5] * torch.sin(theta)
    return z_offset / 2


def junk(self, road_maps):
    P2 = self.P2
    fy = P2[:, 1:2, 1:2] #[B,1, 1]
    cy = P2[:, 1:2, 2:3] #[B,1, 1]
    cx = P2[:, 0:1, 2:3] #[B,1, 1]
    N = self.anchors.shape[1]
    B = road_maps.shape[0]
    if len(self.anchor_mean_std.shape) == 4:
        # B, types, N, (z, sina, cosa, w, h, l), (mean, std)
        self.anchor_mean_std = self.anchor_mean_std.unsqueeze(0).repeat(B, 1, 1, 1, 1)
    anchors_z = self.get_anchors_z_from_3_sides(road_maps) # [B, types, N]
    anchors_z_2 = self.anchor_mean_std[:, :, :, 0, 0].permute(0, 2, 1)
    anchors_z = torch.cat([anchors_z, anchors_z_2], dim=2)

    anchors = torch.cat([self.anchors, self.anchors], dim=1)
    anchor_mean_std = torch.cat([self.anchor_mean_std, self.anchor_mean_std], dim=1)
    anchors_image_x_center = torch.cat(
        [self.anchors_image_x_center, self.anchors_image_x_center], dim=0)
    anchors_image_y_center = torch.cat(
        [self.anchors_image_y_center, self.anchors_image_y_center], dim=0)

    # anchors_z = self.get_anchors_z(road_maps) # [B, types, N]

    # import pdb; pdb.set_trace()
    world_x3d = (anchors_image_x_center * anchors_z -
                 anchors_z.new(cx) * anchors_z) / anchors_z.new(fy) #[B, types, N]
    world_y3d = (anchors_image_y_center * anchors_z -
                 anchors_z.new(cy) * anchors_z) / anchors_z.new(fy) #[B, types, N]

    #[B,N] any one type lies in target range
    useful_mask = torch.any((world_y3d > self.filter_y_threshold_min_max[0]) *
                            (world_y3d < self.filter_y_threshold_min_max[1]) *
                            (world_x3d.abs() < self.filter_x_threshold), dim=1)

    anchors_z = anchors_z.permute(0, 2, 1)
    anchor_mean_std[:, :, :, 0, 0] = anchors_z
    # self.anchor_mean_std[:, :, :, 0, 1] = 1
