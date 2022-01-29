from visualDet3D.evaluator.kitti.evaluate import evaluate_synthia


result_path = '/misc/lmbraid18/shanus/3D_object_detection/synthia/Mono3D_full_new_code/output/validation/data'
label_path = '/home/shanus/datasets/SYNTHIA_ICCV2019/'
label_split_file = '/home/shanus/datasets/SYNTHIA_ICCV2019/splits/testing_split_filtered.json'

print(evaluate_synthia(label_path, result_path, label_split_file)[0])
