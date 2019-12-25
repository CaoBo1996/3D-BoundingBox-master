import cv2
import numpy as np
import os
import random

import torch
from torchvision import transforms
from torch.utils import data

from library.File import *

from .ClassAverages import ClassAverages


# TODO: clean up where this is
def generate_bins(bins):

    # 将整个0-2pi的空间均匀分割成bins部分， 然后返回的是每段的中心角度
    # 对于bins==2,则angle_bins=[pi/2,3pi/2]
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1, bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2  # center of the bin

    return angle_bins


class Dataset(data.Dataset):
    def __init__(self, path, bins=2, overlap=0.1):

        self.top_label_path = path + "label_2" + os.path.sep  # 训练图片标签路径
        self.top_img_path = path + "image_2" + os.path.sep  # 训练图片路径
        self.top_calib_path = path + "calib" + os.path.sep  # 训练相机标定参数路径

        # TODO: which camera cal to use, per frame or global one?
        # 训练的时候不相机标定参数，但为了代码兼容性，传入global proj_matrix
        # 路径为 ''e:\\3DCode\\CODE\\3D-BoundingBox-master\\torch_lib\\camera_cal\\calib_cam_to_cam.txt''
        global_proj_matrix = os.path.abspath(os.path.dirname(__file__) + os.path.sep + 'camera_cal' + os.path.sep + 'calib_cam_to_cam.txt')
        self.proj_matrix = get_P(global_proj_matrix)

        # 取得top_img_path路径下所有图片的名称，KITTI数据集中图片名称是id
        # 000000.png 000001.png ...
        self.ids = [x.split('.')[0] for x in sorted(os.listdir(self.top_img_path))]

        # 图片数量
        self.num_images = len(self.ids)

        # create angle bins
        self.bins = bins

        # [0.,0.]
        self.angle_bins = np.zeros(bins)

        # 根据bins的值将2pi角等分为bins部分，interval为每部分的角大小
        self.interval = 2 * np.pi / bins

        # 得到每一个angle_bin的起始角度值
        for i in range(1, bins):
            self.angle_bins[i] = i * self.interval

        # 每一个bin的center角度，对于bins==2,则为 pi / 2和 3pi/2
        # interval为每等分的abgle_bin值，interval/2即每等分一半的值
        # 然后用每个angle_bin起始点的值加上interval/2的值，得到每个
        # angle_bin的center的角度值，即abgle_bins存放着是每个angle_bin
        # 的center角度
        self.angle_bins += self.interval / 2  # center of the bin

        # 为什么要设置overlap
        self.overlap = overlap

        # [(min angle in bin, max angle in bin), ... ]
        # bin_ranges存放着是每个bin的范围
        self.bin_ranges = []

        # 范围为[2pi-0.1,pi+0.1]和[pi-0.1,0.1]
        # [(6.183185307179587, 3.241592653589793), (3.041592653589793, 0.09999999999999964)]
        # 即这两个bin并没有将整个0-2pi给全部覆盖
        for i in range(0, bins):
            self.bin_ranges.append(((i * self.interval - overlap) % (2 * np.pi), (i * self.interval + self.interval + overlap) % (2 * np.pi)))

        # 统计平均维度信息，即对训练集出现的以下几个类别进行统计平均维度
        # count是出现的次数，total是高，宽，长总的值
        # 注意：class_list包括了KITTI数据集的所有的类别
        class_list = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
        self.averages = ClassAverages(class_list)

        # 返回的是label文件的id和相应的label文件里的行数
        self.object_list = self.get_objects(self.ids)

        self.labels = {}
        last_id = ""

        # 循环遍历object_list
        for obj in self.object_list:
            id = obj[0]
            line_num = obj[1]

            # 得到label内容
            label = self.get_label(id, line_num)
            if id != last_id:
                self.labels[id] = {}
                last_id = id

            self.labels[id][str(line_num)] = label

        # hold one image at a time
        self.curr_id = ""
        self.curr_img = None

    def __getitem__(self, index):
        id = self.object_list[index][0]
        line_num = self.object_list[index][1]

        if id != self.curr_id:
            self.curr_id = id
            self.curr_img = cv2.imread(self.top_img_path + '%s.png' % id)

        label = self.labels[id][str(line_num)]
        # P doesn't matter here
        obj = DetectedObject(self.curr_img, label['Class'], label['Box_2D'], self.proj_matrix, label=label)

        return obj.img, label

    def __len__(self):
        return len(self.object_list)

    # 返回每一个label文件的行数和id并且统计维度信息，将其加入到class_averages.txt文件中
    # 第一个label文件的id为000000,总共有1行
    # 第二个label文件的id为000001,总共有2行
    # [('000000', 0), ('000001', 0), ('000001', 1), ('000002', 0), ('000003', 0), ...]
    def get_objects(self, ids):
        objects = []
        for id in ids:
            with open(self.top_label_path + '%s.txt' % id) as file:

                # 遍历label标签文件的每一行，第一行的line_num为0
                for line_num, line in enumerate(file):
                    line = line[:-1].split(' ')  # 将label标签每一行变成一个list
                    obj_class = line[0]  # 得到类别
                    if obj_class == "DontCare":
                        continue

                    # 得到车的维度
                    dimension = np.array([float(line[8]), float(line[9]), float(line[10])], dtype=np.double)

                    # 将训练集上的图片类别和维度信息加入到class_averages文件中
                    self.averages.add_item(obj_class, dimension)

                    # 返回每一个label文件的id和行数
                    # 注意：对于同一个id，可能对应多个行
                    objects.append((id, line_num))

        # 最终将统计信息给写进去文件中class_averages.txt中
        self.averages.dump_to_file()
        return objects

    # 根据id信息和line_num信息来得到label文件id.txt的第line_num行
    # 然后通过format_label()函数对第line_num行进行格式化
    # lines是一个列表
    def get_label(self, id, line_num):
        lines = open(self.top_label_path + '%s.txt' % id).read().splitlines()

        # lines[line_num]是第line_num行
        label = self.format_label(lines[line_num])

        return label

    def get_bin(self, angle):

        # angle是alpha转换成0-2pi的结果
        bin_idxs = []

        def is_between(min, max, angle):
            max = (max - min) if (max - min) > 0 else (max - min) + 2 * np.pi
            angle = (angle - min) if (angle - min) > 0 else (angle - min) + 2 * np.pi
            return angle < max

        # 对于每个bin，根据其范围，判断angle是否在这个bin范围内
        for bin_idx, bin_range in enumerate(self.bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)

        return bin_idxs

    # 对id.txt的第line_num行内容进行格式化，将标签的信息提取出来
    def format_label(self, line):
        line = line[:-1].split(' ')

        # 得到类别信息
        Class = line[0]

        # 将str转换成float类型
        for i in range(1, len(line)):
            line[i] = float(line[i])

        Alpha = line[3]  # what we will be regressing

        # TODO 并没有用到，但是这里读进来为什么是0.0，000000.txt
        # 中是0.01啊？奇怪
        Ry = line[14]

        # round()函数，返回浮点数的四舍五入的值，根据KITTI数据集的说明
        # TODO 2d box是像素坐标，但是为什么label文件中会出现小数值呢？
        top_left = (int(round(line[4])), int(round(line[5])))
        bottom_right = (int(round(line[6])), int(round(line[7])))

        # [(712, 143), (811, 308)]
        Box_2D = [top_left, bottom_right]

        # height, width, length
        Dimension = np.array([line[8], line[9], line[10]], dtype=np.double)

        # modify for the average
        # 注意在run.py代码中，会将预测到的值再加上平均高宽长
        Dimension -= self.averages.get_item(Class)

        # Location是物体下底面中心在相机坐标系的坐标
        Location = [line[11], line[12], line[13]]  # x, y, z

        # TODO 意思是 KITTI数据集中的location的位置是底面中心的位置？
        # 但是就算是这样，也不应该减去Dimension[0]啊？因为此时的dimension[0]
        # 已经是GT值减去平均值了啊？应该是减去原来的dimension[0]啊？？
        # 但是在训练集中并没有用到这个参数Location参数，所以不影响？
        # Location[1] -= Dimension[0] / 2  # bring the KITTI center up to the middle of the object

        Location[1] -= line[8]  # TODO 我觉得应该是这样计算的

        # array([[0., 0.],
        # [0., 0.]])
        # Orientation存储sin和cos的值，因为有两个bin，所以对于每个
        # bin，都有一个sin和cos，所以总共是4个值
        Orientation = np.zeros((self.bins, 2))

        # 对每一个bin，均对应着一个Confidence
        # 并且每一个Confidence的初始值均为0
        Confidence = np.zeros(self.bins)

        # alpha is [-pi..pi], shift it to be [0..2pi]
        # 对标签文件的alpha进行转换到0-2pi
        angle = Alpha + np.pi

        # 根据我们创建的angle_bin，看alpha是否在某个bin范围内
        # 统计angle在bin中的所有的bin的索引
        bin_idxs = self.get_bin(angle)

        for bin_idx in bin_idxs:

            # 计算angle与落在bin的中点的角度误差
            angle_diff = angle - self.angle_bins[bin_idx]

            # 计算angle_diff的sin和cos的值
            Orientation[bin_idx, :] = np.array([np.cos(angle_diff), np.sin(angle_diff)])

            # 当从alpha从-pi-pi转换成0-2pi的angle，若落在了哪个bin之中
            # 那么就把该bin对应着的Condidence的值设置为1
            # 并且计算此时angle相对于该bin center的角度sin 和cos的值
            Confidence[bin_idx] = 1

        label = {'Class': Class, 'Box_2D': Box_2D, 'Dimensions': Dimension, 'Alpha': Alpha, 'Orientation': Orientation, 'Confidence': Confidence}

        return label


"""
What is *sorta* the input to the neural net. Will hold the cropped image and
the angle to that image, and (optionally) the label for the object. The idea
is to keep this abstract enough so it can be used in combination with YOLO
"""


class DetectedObject:
    def __init__(self, img, detection_class, box_2d, proj_matrix, label=None):

        if isinstance(proj_matrix, str):  # filename
            # proj_matrix = get_P(proj_matrix)
            # 注意：这个函数识别以P2开头的行作为proj_matrix
            proj_matrix = get_calibration_cam_to_image(proj_matrix)

        self.proj_matrix = proj_matrix
        self.theta_ray = self.calc_theta_ray(img, box_2d, proj_matrix)
        self.img = self.format_img(img, box_2d)
        self.label = label
        self.detection_class = detection_class

    # 计算角度
    def calc_theta_ray(self, img, box_2d, proj_matrix):
        width = img.shape[1]
        fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
        center = (box_2d[1][0] + box_2d[0][0]) / 2
        dx = center - (width / 2)

        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        angle = np.arctan((2 * dx * np.tan(fovx / 2)) / width)
        angle = angle * mult

        return angle

    def format_img(self, img, box_2d):

        # Should this happen? or does normalize take care of it. YOLO doesnt like
        # img=img.astype(np.float) / 255

        # torch transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        process = transforms.Compose([transforms.ToTensor(), normalize])

        # crop image
        pt1 = box_2d[0]
        pt2 = box_2d[1]
        crop = img[pt1[1]:pt2[1] + 1, pt1[0]:pt2[0] + 1]
        crop = cv2.resize(src=crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        # recolor, reformat
        batch = process(crop)

        return batch
