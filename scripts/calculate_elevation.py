import cv2
import collections
import json
import math
import numpy as np
import os

# from deepv2d.data_stream.synthia import SynthiaRaw
from visualDet3D.data.synthia.synthiadata import SynthiaCalib


def load_png(filepath):
    return cv2.imread(filepath).astype(np.int)


def load_depth(filepath):
    depth_image = cv2.imread(filepath).astype(np.float)
    red = depth_image[:, :, 2]
    green = depth_image[:, :, 1]
    blue = depth_image[:, :, 0]
    depth = 5000 * (red + green * 256 + blue * 256 * 256) / (256 * 256 * 256 - 1)
    return depth


def get_ground_points(label):
    # label[:300, :] = 2
    # label[301:, :] = 2
    label[:400, :] = 2
    label[401:, :] = 2
    points = np.argwhere(label==1)
    return points


def get_angles(extrinsics):
    rot = extrinsics[:3, :3]
    alpha = math.atan2(rot[2, 1], rot[2, 2])
    beta = math.atan2(rot[2, 0], math.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2))
    gamma = math.atan2(rot[1, 0], rot[0, 0])
    return alpha, beta, gamma


def calculate_y(point, P2, depth, beta):
    z3d = depth[point[0], point[1]]
    cy = P2[1, 2]
    ty = P2[2, 3]
    fy = P2[1, 1]
    y = point[0]
    # print(cy, ' ', fy)
    # print(P2)
    y3d = (y * z3d - cy * z3d - ty) / fy
    # y3d = (y - cy) * (z3d / fy)
    y3d = y3d - (z3d * math.sin(beta))
    # print(z3d)
    return y3d


train_file = '/home/shanus/datasets/SYNTHIA_ICCV2019/splits/training_5k_split.json'
root = '/home/shanus/datasets/SYNTHIA_ICCV2019'

train_lines = []
with open(train_file) as f:
    train_files = json.load(f, object_pairs_hook=collections.OrderedDict)
    for setting in train_files:
        for sample in train_files[setting]:
            train_lines.append(sample)


y3d_list = np.ndarray(10000,)
idx = 0
for train_line in train_lines:
    try:
        calib_path = os.path.join(root, train_line.format('calib_kitti', 'txt'))
        label = load_png(os.path.join(root, train_line.format('SemSeg', 'png')))[:, :, 2]
        depth = load_depth(os.path.join(root, train_line.format('Depth', 'png')))
        # depth = load_depth(train_line.format('Depth', 'png'))
        information_dict = json.load(open(os.path.join(root, sample.format('Information', 'json'))))
        extrinsics = np.array(information_dict['extrinsic']['matrix']).reshape(4, 4).astype(np.float)
        intrinsics = np.array(information_dict['intrinsic']['matrix']).reshape(4, 4).astype(np.float)
        calib = SynthiaCalib(calib_path).read_calib_file()
        points = get_ground_points(label)
        # print(label.shape)
        for point in points:
            # print(point)
            alpha, beta, gamma = get_angles(extrinsics)
            # print(beta)
            # print(intrinsics)
            # print(calib.P2)
            y3d = calculate_y(point, calib.P2, depth, beta)
            y3d_list[idx] = y3d
            print(y3d)
            idx += 1
            break
            # if int(idx % 100):
            #     break
            # if idx == len(y3d_list):
            #     break
        if idx == len(y3d_list):
            break
    except Exception:
        import pdb
        pdb.set_trace()
# print(np.mean(y3d_list))
# print(np.std(y3d_list))
