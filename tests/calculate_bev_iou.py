import cv2
import collections
import json
import os
import numpy as np
import torch

import visualDet3D.utils.utils as utils

from visualDet3D.evaluator.kitti.kitti_common import get_label_annos, get_label_anno
from visualDet3D.evaluator.kitti.eval import get_official_eval_result
from visualDet3D.utils.visualize_utils import draw_points
from visualDet3D.data.synthia.synthiadata import SynthiaCalib, SynthiaLabel
from visualDet3D.networks.utils import BBox3dProjector, BackProjection
from ptools.vis import np2d, np3d

from mean_average_precision import MetricBuilder


projector = BBox3dProjector()
# backprojector = BackProjection()


def get_gt_annos(label_path, label_split_file):
    gt_annos = []
    seq_indices = {}
    with open(label_split_file) as f:
        training_dict = json.load(f, object_pairs_hook=collections.OrderedDict)
        train_lines = []
        start_idx = 0
        end_idx = 0
        for key in training_dict.keys():
            start_idx = end_idx
            for sample in training_dict[key]:
                train_lines.append(sample.strip())
                end_idx += 1
            seq_indices[key] = [start_idx, end_idx]
        for line in train_lines:
            line = line.strip()
            label_file = os.path.join(label_path, line.format('labels_kitti', 'txt'))
            gt_annos.append(get_label_anno(label_file))
    return gt_annos, seq_indices


def filter_indices(split_file, sequences):
    files_dict = json.load(open(split_file), object_pairs_hook=collections.OrderedDict)
    indices = []
    index = 0
    files = []
    for seq in files_dict:
        for filepath in files_dict[seq]:
            seq_to_check = '/'.join(seq.split('/')[5:])
            # import pdb
            # pdb.set_trace()
            if seq_to_check in sequences:
                indices.append(index)
                files.append(filepath)
            index += 1
    return indices, files


def add_bevboxes_to_image(bboxes, shape=(800, 800, 3), color=(0, 0, 0), image=None):
    if type(color) == tuple:
        color = [color] * len(bboxes)
    mean_x, mean_z = 0, 0
    for idx, points in enumerate(bboxes):
        mean_x += np.abs(points[:, :, 0].mean()).astype(np.int)
        mean_z += np.abs(points[:, :, 1].mean()).astype(np.int)
    mean_x = int(mean_x / len(bboxes))
    mean_z = int(mean_z / len(bboxes))
    for idx, points in enumerate(bboxes):
        image = np.ones(shape).astype(np.uint8) * 255 if image is None else image
        points[:, :, 0] += int(shape[0] / 2) - mean_x
        points[:, :, 1] += int(shape[1] / 2) - mean_z
        for bbox in points:
            image = cv2.polylines(image, [bbox.astype(np.int32)], True, color[idx], 1)
    return image


def get_bboxes(bboxes_3d, P2):
    points = []
    if not np.any(bboxes_3d):
        return (np.ones((0, 4, 2)), np.ones((0, 8, 3)))
    # abs_corners, homo_coord, thetas = projector(torch.tensor(bboxes_3d), torch.tensor(P2))
    bboxes_3d[:, :6] = bboxes_3d[:, :6] * 10
    abs_corners, _, _ = projector(torch.tensor(bboxes_3d), torch.tensor(P2))
    points = abs_corners.numpy()[:, (0, 1, 4, 5), :][:, :, (0, 2)]
    return points


def get_3d_bbox(labels):
    bbox_3d = []
    for label in labels:
        bbox_3d.append([label.x, label.y - 0.5 * label.h, label.z, label.w, label.h, label.l, label.alpha])
    bbox_3d = np.array(bbox_3d)
    return bbox_3d


def evaluate_synthia(label_path="/home/hins/Desktop/M3D-RPN/data/kitti/training/label_2",
                     result_path="/home/hins/IROS_try/pytorch-retinanet/output/validation/data",
                     gac_result_path="/home/hins/IROS_try/pytorch-retinanet/output/validation/data",
                     label_split_file="val.txt",
                     current_classes=[0],
                     gpu=0,
                     sequences=[]):
    root = '/home/shanus/datasets/SYNTHIA_ICCV2019/'

    indices, files = filter_indices(label_split_file, sequences)
    idx = 0

    metric_fn_my = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
    metric_fn_gac = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)

    for index, filepath in zip(indices, files):
        result_file = f'{index:06}' + '.txt'

        my_result_file = os.path.join(result_path, result_file)
        gac_result_file = os.path.join(gac_result_path, result_file)
        gt_labels = SynthiaLabel(os.path.join(root, filepath.format('labels_kitti', 'txt'))).read_label_file().data
        my_labels = SynthiaLabel(my_result_file).read_label_file().data
        gac_labels = SynthiaLabel(gac_result_file).read_label_file().data
        my_bbox_3d = get_3d_bbox(my_labels)
        gac_bbox_3d = get_3d_bbox(gac_labels)
        gt_bbox_3d = get_3d_bbox(gt_labels)
        P2 = SynthiaCalib(os.path.join(root, filepath).format('calib_kitti', 'txt')).read_calib_file().P2
        bev_bboxes = [get_bboxes(bbox_3d, P2) for bbox_3d in [my_bbox_3d, gac_bbox_3d, gt_bbox_3d]]
        bev_bboxes = process_for_evaluation(bev_bboxes)
        my_bev_bboxes, gac_bev_bboxes, gt_bev_bboxes = bev_bboxes

        metric_fn_my.add(my_bev_bboxes, gt_bev_bboxes)
        metric_fn_gac.add(gac_bev_bboxes, gt_bev_bboxes)
    ap_my = metric_fn_my.value(iou_thresholds=0.7, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']
    ap_gac = metric_fn_gac.value(iou_thresholds=0.7, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']
    print(ap_my)
    print(ap_gac)

def process_for_evaluation(bev_bboxes):
    import pdb
    pdb.set_trace()

def load_json_accuracy():
    accuracies_file = json.load(open('accuracies.json'))
    gac_accuracy = accuracies_file['gac_accuracy']
    accuracy = accuracies_file['gac_accuracy']
    seqs = accuracies_file['seqs']


if __name__ == '__main__':
    sequences = [
        # 'test/test5_23segs_weather_0_spawn_1_roadTexture_1_P_None_C_None_B_None_WC_None/24-10-2018_20-41-54',
        'test/test5_22segs_weather_0_spawn_2_roadTexture_2_P_None_C_None_B_None_WC_None/24-10-2018_23-00-15',
        # 'test/test5_18segs_weather_4_spawn_1_roadTexture_1_P_None_C_None_B_None_WC_None/24-10-2018_22-23-54',

        # 'test/test5_14segs_weather_2_spawn_1_roadTexture_1_P_None_C_None_B_None_WC_None/24-10-2018_21-27-06',
        # 'test/test5_22segs_weather_0_spawn_2_roadTexture_2_P_None_C_None_B_None_WC_None/24-10-2018_23-00-15',
        # 'test/test5_14segs_weather_2_spawn_0_roadTexture_1_P_None_C_None_B_None_WC_None/24-10-2018_21-05-15'
                 ]

    label_path = '/home/shanus/datasets/SYNTHIA_ICCV2019/'

    gac_result_path = '/misc/lmbraid18/shanus/3D_object_detection/synthia/Mono3D_full_baseline/output/validation/data/'
    # result_path = '/misc/lmbraid18/shanus/3D_object_detection/synthia/Mono3D_full_new_code/output/validation/data'
    result_path = '/misc/lmbraid18/shanus/3D_object_detection/synthia/Mono3D_DeepV2D/output/validation/data/'
    label_split_file = '/home/shanus/datasets/SYNTHIA_ICCV2019/splits/testing_split_filtered.json'

    evaluate_synthia(label_path, result_path, gac_result_path, label_split_file, sequences=sequences)
