import collections
import json
import numpy as np
from PIL import Image


def convert_semantic_map_to_ground_map(semantic_map,
                                       ground_indices=[2]):
    """
    returns ground_map: ndarray
    0 means non-ground and 1 means ground
    """
    ground_map = semantic_map.copy()
    ground_map[np.isin(semantic_map, ground_indices)] = 100
    ground_map[ground_map != 100] = 0
    ground_map[ground_map == 100] = 1
    return ground_map


def sample_depth_map(depth_map, ground_map, params):
    """
    uses full depth_map and ground_map to calculate depth_map just for the ground_map.
    returns depth_map_sample: ndarray
    """
    width, height, crop_top = params

    depth_map[ground_map == 0] = 0
    avg_depth_map = depth_map.sum(axis=1) / np.count_nonzero(ground_map, axis=1)
    avg_depth_map = avg_depth_map.repeat(width).reshape(height, width)
    np.putmask(avg_depth_map, ground_map == 1, depth_map)
    avg_depth_map[np.isnan(avg_depth_map)] = depth_map.max() + 1
    avg_depth_map[:crop_top, :] = depth_map.max() + 1

    return avg_depth_map


def calculate_road_depth(depth_map, semantic_map, dataset):

    # set the size of the semantic map equal to the depth map.
    semantic_map = np.array(
        Image.fromarray(
            semantic_map.astype(np.uint8)
        ).resize((depth_map.shape[::-1]), Image.NEAREST))

    if dataset == 'synthia':
        ground_map = convert_semantic_map_to_ground_map(semantic_map, ground_indices=[1, 2])
        params = (640, 480, 200)
    elif dataset == 'kitti':
        ground_map = convert_semantic_map_to_ground_map(semantic_map, ground_indices=[0])
        params = (1088, 300, 100)
    else:
        raise KeyError('Dataset not found.')

    return sample_depth_map(depth_map, ground_map, params)
