'''
File Created: Sunday, 17th March 2019 3:58:52 pm
Author: Peng YUN (pyun@ust.hk)
Copyright 2018 - 2019 RAM-Lab, RAM-Lab


Orginally designed for Kitti. Modified for Synthia.
'''
import cv2
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from ..kitti.utils import read_image, read_pc_from_bin, _lidar2leftcam, _leftcam2lidar, _leftcam2imgplane


# SYNTHIA
class SynthiaCalib:
    '''
    class storing Synthia calib data
        self.data(None/dict):keys: 'P0', 'P1', 'P2', 'P3', 'R0_rect', 'Tr_velo_to_cam', 'Tr_imu_to_velo'
        self.R0_rect(np.array):  [4,4]
        self.Tr_velo_to_cam(np.array):  [4,4]
    '''

    def __init__(self, calib_path):
        self.path = calib_path
        self.data = None

    def read_calib_file(self):
        '''
        read Synthia calib file
        '''
        calib = dict()
        with open(self.path, 'r') as f:
            str_list = f.readlines()
        str_list = [itm.rstrip() for itm in str_list if itm != '\n']
        for itm in str_list:
            calib[itm.split(':')[0]] = itm.split(':')[1]
        for k, v in calib.items():
            calib[k] = [float(itm) for itm in v.split()]
        self.data = calib

        self.P2 = np.array(self.data['P2']).reshape(3, 4)
        self.P3 = np.array(self.data['P3']).reshape(3, 4)

        R0_rect = np.zeros([4, 4])
        R0_rect[0:3, 0:3] = np.array(self.data['R0_rect']).reshape(3, 3)
        R0_rect[3, 3] = 1
        self.R0_rect = R0_rect


        return self

    def leftcam2lidar(self, pts):
        '''
        transform the pts from the left camera frame to lidar frame
        pts_lidar  = Tr_velo_to_cam^{-1} @ R0_rect^{-1} @ pts_cam
        inputs:
            pts(np.array): [#pts, 3]
                points in the left camera frame
        '''
        if self.data is None:
            print("read_calib_file should be read first")
            raise RuntimeError
        return _leftcam2lidar(pts, self.Tr_velo_to_cam, self.R0_rect)

    def lidar2leftcam(self, pts):
        '''
        transform the pts from the lidar frame to the left camera frame
        pts_cam = R0_rect @ Tr_velo_to_cam @ pts_lidar
        inputs:
            pts(np.array): [#pts, 3]
                points in the lidar frame
        '''
        if self.data is None:
            print("read_calib_file should be read first")
            raise RuntimeError
        return _lidar2leftcam(pts, self.Tr_velo_to_cam, self.R0_rect)

    def leftcam2imgplane(self, pts):
        '''
        project the pts from the left camera frame to left camera plane
        pixels = P2 @ pts_cam
        inputs:
            pts(np.array): [#pts, 3]
            points in the left camera frame
        '''
        if self.data is None:
            print("read_calib_file should be read first")
            raise RuntimeError
        return _leftcam2imgplane(pts, self.P2)


class SynthiaLabel:
    '''
    class storing Synthia 3d object detection label
        self.data ([SynthiaObj])
    '''

    def __init__(self, label_path=None):
        self.path = label_path
        self.data = None

    def read_label_file(self, no_dontcare=True):
        '''
        read Synthia label file
        '''
        self.data = []
        with open(self.path, 'r') as f:
            str_list = f.readlines()
        str_list = [itm.rstrip() for itm in str_list if itm != '\n']
        for s in str_list:
            self.data.append(SynthiaObj(s))
        if no_dontcare:
            self.data = list(filter(lambda obj: obj.type != "DontCare", self.data))
        return self

    def __str__(self):
        '''
        TODO: Unit TEST
        '''
        s = ''
        for obj in self.data:
            s += obj.__str__() + '\n'
        return s

    def equal(self, label, acc_cls, rtol):
        '''
        equal oprator for SynthiaLabel
        inputs:
            label: SynthiaLabel
            acc_cls: list [str]
                ['Car', 'Van']
            eot: float
        Notes: O(N^2)
        '''
        if len(self.data) != len(label.data):
            return False
        if len(self.data) == 0:
            return True
        bool_list = []
        for obj1 in self.data:
            bool_obj1 = False
            for obj2 in label.data:
                bool_obj1 = bool_obj1 or obj1.equal(obj2, acc_cls, rtol)
            bool_list.append(bool_obj1)
        return any(bool_list)

    def isempty(self):
        '''
        return True if self.data = None or self.data = []
        '''
        return self.data is None or len(self.data) == 0


class SynthiaObj():
    '''
    class storing a Synthia 3d object
    '''

    def __init__(self, s=None):
        self.type = None
        self.truncated = None
        self.occluded = None
        self.alpha = None
        self.bbox_l = None
        self.bbox_t = None
        self.bbox_r = None
        self.bbox_b = None
        self.h = None
        self.w = None
        self.l = None
        self.x = None
        self.y = None
        self.z = None
        self.ry = None
        self.score = None
        if s is None:
            return
        if len(s.split()) == 15:  # data
            self.truncated, self.occluded, self.alpha,\
                self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b, \
                self.h, self.w, self.l, self.x, self.y, self.z, self.ry = \
                [float(itm) for itm in s.split()[1:]]
            self.type = s.split()[0]
        elif len(s.split()) == 16:  # result
            self.truncated, self.occluded, self.alpha,\
                self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b, \
                self.h, self.w, self.l, self.x, self.y, self.z, self.ry, self.score = \
                [float(itm) for itm in s.split()[1:]]
            self.type = s.split()[0]
        else:
            raise NotImplementedError

    def __str__(self):
        if self.score is None:
            return "{} {:.2f} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
                self.type, self.truncated, int(self.occluded), self.alpha,
                self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b,
                self.h, self.w, self.l, self.x, self.y, self.z, self.ry)
        else:
            return "{} {:.2f} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
                self.type, self.truncated, int(self.occluded), self.alpha,
                self.bbox_l, self.bbox_t, self.bbox_r, self.bbox_b,
                self.h, self.w, self.l, self.x, self.y, self.z, self.ry, self.score)


def load_depth(filepath):
    depth_image = cv2.imread(filepath).astype(np.float)
    red = depth_image[:, :, 2]
    green = depth_image[:, :, 1]
    blue = depth_image[:, :, 0]
    depth = 5000 * (red + green * 256 + blue * 256 * 256) / (256 * 256 * 256 - 1)
    return depth


class SynthiaData:
    '''
    class storing a frame of Synthia data
    '''

    def __init__(self, root_dir, placeholder_path, output_dict=None):
        '''
        inputs:
            root_dir(str): Synthia dataset dir
            idx(str %6d): data index e.g. "000000"
            output_dict: decide what to output
        '''
        self.calib_path = os.path.join(root_dir, placeholder_path.format("calib_kitti", "txt"))
        self.image2_path = os.path.join(root_dir, placeholder_path.format("RGB", "png"))
        self.image3_path = os.path.join(root_dir, placeholder_path.format("RGB", "png"))
        self.label2_path = os.path.join(root_dir, placeholder_path.format("labels_kitti", "txt"))
        self.depth_path = os.path.join(root_dir, placeholder_path.format("Depth", "png"))
        self.semseg_path = os.path.join(root_dir, placeholder_path.format("SemSeg", "png"))
        # self.road_map_path = os.path.join(root_dir, placeholder_path.format("DeepV2D_RoadMap", "npy"))
        self.road_map_path = os.path.join(root_dir, placeholder_path.format("RoadMap", "npy"))
        self.output_dict = output_dict
        if self.output_dict is None:
            self.output_dict = {
                "calib": True,
                "image": True,
                "image_3": False,
                "label": True,
                "velodyne": False,
                "road_map": True
            }

    def read_depth_semseg(self):
        '''
        read depth information
        '''
        return (load_depth(self.depth_path), cv2.imread(self.semseg_path)[:, :, 2])

    def get_masked_road_map(self):
        ground_indices = [1, 20]
        semseg_map = self.read_depth_semseg()[1]
        semseg_map[np.isin(semseg_map, ground_indices)] = 100
        semseg_map[semseg_map != 100] = 0
        semseg_map[semseg_map == 100] = 1
        road_map = np.load(self.road_map_path)
        road_map[semseg_map == 0] = 0
        return road_map

    def gaussian_smooth_road_map(self):
        road_map = np.load(self.road_map_path)
        return gaussian_filter(road_map, sigma=[3, 10])

    def read_road_map(self):
        road_map = np.load(self.road_map_path)
        return road_map

    def read_data(self):
        '''
        read data
        returns:
            calib(SynthiaCalib)
            image(np.array): [w, h, 3]
            label(SynthiaLabel)
            pc(np.array): [# of points, 4]
                point cloud in lidar frame.
                [x, y, z]
                      ^x
                      |
                y<----.z
        '''

        calib = SynthiaCalib(self.calib_path).read_calib_file() if self.output_dict["calib"] else None
        image = read_image(self.image2_path) if self.output_dict["image"] else None
        road_map = self.gaussian_smooth_road_map() if self.output_dict["road_map"] else None
        image = image[:, :, :3] if image is not None else None
        label = SynthiaLabel(self.label2_path).read_label_file() if self.output_dict["label"] else None
        pc = read_pc_from_bin(self.velodyne_path) if self.output_dict["velodyne"] else None
        if 'image_3' in self.output_dict and self.output_dict['image_3']:
            image_3 = read_image(self.image3_path) if self.output_dict["image_3"] else None
            return calib, image, image_3, label, pc, road_map
        else:
            return calib, image, label, pc, road_map
