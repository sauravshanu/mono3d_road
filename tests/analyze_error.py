import collections
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from visualDet3D.evaluator.kitti.kitti_common import get_label_annos, get_label_anno
from visualDet3D.evaluator.kitti.eval import get_official_eval_result
from numba import cuda


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


def evaluate_synthia(label_path="/home/hins/Desktop/M3D-RPN/data/kitti/training/label_2",
                     result_path="/home/hins/IROS_try/pytorch-retinanet/output/validation/data",
                     gac_result_path="/home/hins/IROS_try/pytorch-retinanet/output/validation/data",
                     label_split_file="val.txt",
                     current_classes=[0],
                     gpu=0):
    cuda.select_device(gpu)
    dt_annos = get_label_annos(result_path)
    gac_dt_annos = get_label_annos(gac_result_path)
    gt_annos, seq_indices = get_gt_annos(label_path, label_split_file)

    accuracy_file_name = 'accuracies_gac_deepv2d_without_gac.json'
    result_texts = []
    accuracies = []
    gac_accuracies = []
    seqs = []
    accuracy_dict = {}
    for current_class in current_classes:
        # result_text = get_official_eval_result(gt_annos, dt_annos, current_class)
        # print(result_text)
        # return
        for seq, indices in seq_indices.items():
            start_idx, end_idx = indices
            gt_anno = gt_annos[start_idx: end_idx]
            dt_anno = dt_annos[start_idx: end_idx]
            gac_dt_anno = gac_dt_annos[start_idx: end_idx]
            assert len(gt_anno) == len(dt_anno)
            if gt_anno == []:
                continue
            try:
                result_text = get_official_eval_result(gt_anno, dt_anno, current_class)
                gac_result_text = get_official_eval_result(gt_anno, gac_dt_anno, current_class)
                # print('Sequence is {}.'.format(seq))
                # print(result_text)
                # print(gac_result_text)
            except Exception as e:
                # import pdb
                # pdb.set_trace()
                print(e)
            accuracy = parse_result_text(result_text)
            gac_accuracy = parse_result_text(gac_result_text)
            accuracy_dict[seq] = [float(accuracy), float(gac_accuracy)]
            accuracies.append(accuracy)
            gac_accuracies.append(gac_accuracy)
            seqs.append(seq)
        json.dump(
            dict(accuracy=accuracies, gac_accuracy=gac_accuracies, seqs=seqs),
            open(accuracy_file_name, 'w'),
            indent=4)
        # plot(accuracy_dict, accuracy_file_name)
        # print('My accuracy for {} sequence is {}'.format(seq, accuracy))
        # print('GAC accuracy for {} sequence is {}'.format(seq, gac_accuracy))
    # return result_texts


def plot(accuracy_dict, accuracy_file_name):
    seqs = accuracy_dict.keys()
    accuracy = [accuracy[0] for accuracy in accuracy_dict.values()]
    gac_accuracy = [accuracy[1] for accuracy in accuracy_dict.values()]
    x = np.arange(len(seqs))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, accuracy, width, label='my')
    rects2 = ax.bar(x + width/2, gac_accuracy, width, label='gac')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('GAC vs My accuracies')
    ax.set_xticks(x)
    # ax.set_xticklabels(seqs)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    # plt.show()
    accuracy_file_name = accuracy_file_name.replace('json', 'png')
    plt.savefig(accuracy_file_name)


def parse_result_text(result_text):
    lines = result_text.splitlines()
    acc_3d = lines[3].split()[1].strip()[3:-1]
    return float(acc_3d)


def load_json_accuracy():
    accuracies_file = json.load(open('accuracies.json'))
    gac_accuracy = accuracies_file['gac_accuracy']
    accuracy = accuracies_file['gac_accuracy']
    seqs = accuracies_file['seqs']
    plot(seqs, gac_accuracy, accuracy)


# load_json_accuracy()

label_path = '/home/shanus/datasets/SYNTHIA_ICCV2019/'

gac_result_path = '/misc/lmbraid18/shanus/3D_object_detection/synthia/Mono3D_full_baseline/output/validation/data/'
result_path = '/misc/lmbraid18/shanus/3D_object_detection/synthia/Mono3D_full_new_code/output/validation/data/'
label_split_file = '/home/shanus/datasets/SYNTHIA_ICCV2019/splits/testing_split_filtered.json'

# label_split_file = '/home/shanus/datasets/SYNTHIA_ICCV2019/splits/testing_10k_split.json'
# gac_result_path = '/misc/lmbraid18/shanus/3D_object_detection/synthia/Mono3D_10k_baseline/output/validation/data/'
# result_path = '/misc/lmbraid18/shanus/3D_object_detection/synthia/Mono3D_10k_new_code/output/validation/data/'

# label_split_file = '/home/shanus/datasets/SYNTHIA_ICCV2019/splits/testing_10k_split.json'
# gac_result_path = '/misc/lmbraid18/shanus/3D_object_detection/synthia/Mono3D_10k_baseline_trained_on_full/output/validation/data/'
# result_path = '/misc/lmbraid18/shanus/3D_object_detection/synthia/Mono3D_10k_my_trained_on_full/output/validation/data/'

gac_result_path = '/misc/lmbraid18/shanus/3D_object_detection/synthia/Mono3D_full_baseline/output/validation/data/'
result_path = '/misc/lmbraid18/shanus/3D_object_detection/synthia/Mono3D_full_new_code/output/validation/data'
result_path = '/misc/lmbraid18/shanus/3D_object_detection/synthia/Mono3D_DeepV2D/output/validation/data/'
label_split_file = '/home/shanus/datasets/SYNTHIA_ICCV2019/splits/testing_split_filtered.json'

evaluate_synthia(label_path, result_path, gac_result_path, label_split_file)
