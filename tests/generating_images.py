import collections
import cv2
import json
import numpy as np
import os
from PIL import Image

from ptools.vis import np2d, np3d
from visualDet3D.utils.imdb import IMDB

disk_dict = {
    '5H': {
        'train_split': '/home/shanus/datasets/SYNTHIA_ICCV2019/splits/training_5H_split.json',
        'val_split': '/home/shanus/datasets/SYNTHIA_ICCV2019/splits/testing_5H_split.json',
        'depth_train' : '/misc/lmbraid18/shanus/datasets/SYNTHIA_ICCV2019/depth/5H/train',
        'depth_test' : '/misc/lmbraid18/shanus/datasets/SYNTHIA_ICCV2019/depth/5H/val',
        'semantic_train' : '/misc/lmbraid18/shanus/datasets/SYNTHIA_ICCV2019/ss_output/5H/train',
        'semantic_test' : '/misc/lmbraid18/shanus/datasets/SYNTHIA_ICCV2019/ss_output/5H/val',
        'road_map_train' : '/misc/lmbraid18/shanus/datasets/SYNTHIA_ICCV2019/road_map/5H/train',
        'road_map_val' : '/misc/lmbraid18/shanus/datasets/SYNTHIA_ICCV2019/road_map/5H/val',
    },
    '5k': {
        'depth_train' : '/misc/lmbraid18/shanus/datasets/SYNTHIA_ICCV2019/depth/5k/train',
        'depth_test' : '/misc/lmbraid18/shanus/datasets/SYNTHIA_ICCV2019/depth/5k/val',
        'semantic_train' : '/misc/lmbraid18/shanus/datasets/SYNTHIA_ICCV2019/ss_output/5k/train',
        'semantic_test' : '/misc/lmbraid18/shanus/datasets/SYNTHIA_ICCV2019/ss_output/5k/val',
        'road_map_train' : '/misc/lmbraid18/shanus/datasets/SYNTHIA_ICCV2019/road_map/5k/train',
        'road_map_val' : '/misc/lmbraid18/shanus/datasets/SYNTHIA_ICCV2019/road_map/5k/val',
    }
}


def load_method(filepath):
    return np.load(filepath + '.npy')


def get_road_map(index, split):
    filepath = os.path.join(disk_dict[split]['road_map_train'], str(index) + '.npy')
    return np.load(filepath)


def get_ss_map(filepath):
    ss_image = cv2.imread(filepath).astype(np.float)[:, :, 2]
    return ss_image


def get_depth_map(filepath):
    depth_image = cv2.imread(filepath).astype(np.float)
    red = depth_image[:, :, 2]
    green = depth_image[:, :, 1]
    blue = depth_image[:, :, 0]
    depth = 5000 * (red + green * 256 + blue * 256 * 256) / (256 * 256 * 256 - 1)
    return depth


def get_image(index, split):
    split_file = disk_dict[split]['train_split']
    train_lines = []
    with open(split_file) as f:
        train_files = json.load(f, object_pairs_hook=collections.OrderedDict)
        for setting in train_files:
            for sample in train_files[setting]:
                sample = os.path.join('/home/shanus/datasets/SYNTHIA_ICCV2019/', sample)
                train_lines.append(sample)
    sample = train_lines[index]
    depth_image = get_depth_map(sample.format('Depth', 'png'))
    ss_image = get_ss_map(sample.format('SemSeg', 'png'))
    image = cv2.imread(sample.format('RGB', 'png')).astype(np.float)
    return image, ss_image, depth_image


def save_image(image, index, split, img_type):
    filepath = '/home/shanus/workspace/3D_detection/visualDet3D/images/{}_{}_{}.png'.format(split, index, img_type)
    if len(image.shape) == 3:
        image_obj = np3d(image, image_range_text_off=True)
    else:
        image_obj = np2d(image, image_range_text_off=True)
    image_obj.save(filepath)


def write_images(split):
    index = 211
    road_map = get_road_map(index, split)
    # ss_map = get_ss_map(index, split)
    # depth_map = get_depth_map(index, split)
    image, ss_map, depth_map = get_image(index, split)
    images = [(road_map, 'road_map'), (ss_map, 'ss_map'), (depth_map, 'depth_map'), (image, 'image')]
    for image, img_type in images:
        save_image(image, index, split, img_type)


write_images('5H')
# write_images('5k')
