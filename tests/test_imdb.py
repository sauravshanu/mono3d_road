import collections
import json
import sys
# sys.path.append('/misc/student/shanus/workspace/3D_detection/visualDet3D/')

from visualDet3D.utils.imdb import IMDB
from visualDet3D.data.synthia.synthiadata import SynthiaData


frames_size = 50


def read_one(split_file, imdb_path):
    output_dict = {
        "calib": True,
        "image": True,
        "label": True,
        "velodyne": False,
    }
    final_index = 0
    original_train_files = []
    with open(split_file) as f:
        imdb = IMDB(imdb_path, 100)
        print(len(imdb._frames))
        train_files = json.load(f, object_pairs_hook=collections.OrderedDict)
        train_lines = []
        for setting in train_files:
            for sample in train_files[setting]:
                sample = SynthiaData(imdb_path, sample, output_dict)
                imdb.append(sample)
                original_train_files.append(sample.image2_path)
                final_index += 1
        imdb.save()


imdb_path = '/misc/lmbraid18/shanus/3D_object_detection/synthia/tmp_dir'
train_file = '/home/shanus/datasets/SYNTHIA_ICCV2019/splits/training_5H_split.json'
val_file = '/home/shanus/datasets/SYNTHIA_ICCV2019/splits/testing_5H_split.json'
read_one(train_file, imdb_path)
read_one(val_file, imdb_path)

# imdb = IMDB.load_from_disk(imdb_path)
# expected_train_files = []
# for idx in range(len(imdb)):
#     # print(imdb[idx])
#     expected_train_files.append(imdb[idx].image2_path)
# print(final_index)
# print(len(imdb))

# assert len(imdb) == final_index
# assert original_train_files == expected_train_files
