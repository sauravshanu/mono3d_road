import collections
import json
import os

from .kitti_common import get_label_annos, get_label_anno
from .eval import get_official_eval_result
from numba import cuda


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(label_path="/home/hins/Desktop/M3D-RPN/data/kitti/training/label_2",
             result_path="/home/hins/IROS_try/pytorch-retinanet/output/validation/data",
             label_split_file="val.txt",
             current_classes=[0],
             gpu=0):
    cuda.select_device(gpu)
    dt_annos = get_label_annos(result_path)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = get_label_annos(label_path, val_image_ids)
    result_texts = []
    for current_class in current_classes:
        result_texts.append(get_official_eval_result(gt_annos, dt_annos, current_class))
    return result_texts


def evaluate_synthia(label_path="/home/hins/Desktop/M3D-RPN/data/kitti/training/label_2",
                     result_path="/home/hins/IROS_try/pytorch-retinanet/output/validation/data",
                     label_split_file="val.txt",
                     current_classes=[0],
                     gpu=0):
    cuda.select_device(gpu)
    dt_annos = get_label_annos(result_path)
    # val_image_ids = _read_imageset_file(label_split_file)
    # read labels from synthia dataset.
    # synthia files are stored in much different structure than kitti so we
    # need this.
    gt_annos = []
    with open(label_split_file) as f:
        training_dict = json.load(f, object_pairs_hook=collections.OrderedDict)
        idx = 0
        train_lines = []
        for key in training_dict.keys():
            for sample in training_dict[key]:
                train_lines.append(sample.strip())
        for line in train_lines:
            line = line.strip()
            label_file = os.path.join(label_path, line.format('labels_kitti', 'txt'))
            gt_annos.append(get_label_anno(label_file))
    # gt_annos = get_label_annos(label_path, val_image_ids)
    result_texts = []
    for current_class in current_classes:
        result_texts.append(get_official_eval_result(gt_annos, dt_annos, current_class))
    return result_texts
