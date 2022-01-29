import cv2
import numpy as np
import os
import tqdm

import road_map_utils


def process_split_file(split_file):

    samples_lines = []
    with open(split_file) as f:
        samples_lines = f.readlines()
        for i in range(len(samples_lines)):
            samples_lines[i] = samples_lines[i].strip()
    return samples_lines


def generate_road_map_gt(split_file, depth_file, ss_root, road_root):
    print('Processing split file {}.'.format(split_file))
    samples = process_split_file(split_file)
    depth = np.load(depth_file)
    for idx, sample in tqdm.tqdm(enumerate(samples)):
        depth_sample = depth[idx]
        ss_output_sample = np.load(os.path.join(ss_root, sample + '.png.npy'))
        road_map = road_map_utils.calculate_road_depth(depth_sample, ss_output_sample, 'kitti')
        np.save(os.path.join(road_root, sample + '.npy'), road_map)


if __name__ == '__main__':
    train_split_file = '/home/shanus/workspace/3D_detection/visualDet3D/visualDet3D/data/kitti/chen_split/train.txt'
    train_depth_file = '/misc/lmbraid18/shanus/datasets/KITTI/3D_object_detection/depth_train.npy'
    ss_root = '/misc/lmbraid18/shanus/datasets/KITTI/3D_object_detection/ss_nvidia/train'
    road_root = '/misc/lmbraid18/shanus/datasets/KITTI/3D_object_detection/road_map/train'

    val_split_file = '/home/shanus/workspace/3D_detection/visualDet3D/visualDet3D/data/kitti/chen_split/val.txt'
    val_depth_file = '/misc/lmbraid18/shanus/datasets/KITTI/3D_object_detection/depth_val.npy'
    generate_road_map_gt(train_split_file, train_depth_file, ss_root, road_root)
    generate_road_map_gt(val_split_file, val_depth_file, ss_root, road_root)
