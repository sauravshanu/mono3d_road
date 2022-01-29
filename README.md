# Monocular 3D Object Detection in Driving Scenarios using the Road Information:

This repo provides training and evaluation code for monocular 3D object detection. It generates road depth map using the semantic segmentation map and depth map of the 
scene and then it uses the road depth map to sample better 3D anchors and feeds it to the network for 3D bounding box prediction.

Major backbone of this code is borrowed from very well structed [VisualDet3D](https://github.com/Owen-Liuyuxuan/visualDet3D) code. 

### Here's the overview of this unique method.

<img src='https://raw.githubusercontent.com/saurav1869/mono3d_road/master/network_design.png'>

<hr/>

### Here are the predictions compared to the ground truth(in green).

<img src="https://github.com/saurav1869/saurav1869/blob/main/movie_short.gif" width="400"><img src="https://github.com/saurav1869/saurav1869/blob/main/bev_movie_short.gif" width="400">

## Setup 

```bash
pip3 install -r requirement.txt
```


```bash
# build ops (deform convs and iou3d), We will not install operations into the system environment
./make.sh
```

## Training
By default trains for synthia. For KITTI change `config/Road_synthia_train_full.py`.
```bash
./launcher/det_precompute.sh config/Road_synthia_train_full.py road
./launcher/train.sh  --config/Road_synthia_train_full.py 0 mono3d_training_v1
```
