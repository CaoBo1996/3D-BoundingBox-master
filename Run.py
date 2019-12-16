"""
Images must be in ./Kitti/testing/image_2/ and camera matricies in ./Kitti/testing/calib/

Uses YOLO to obtain 2D box, PyTorch to get 3D box, plots both

SPACE bar for next image, any other key to exit
"""

from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages
from yolo.yolo import cv_Yolo

import os
import time

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg

import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument("--image-dir", default="eval/image_2/", help="Relative path to the directory containing images to detect. Default \
                    is eval/image_2/")

# TODO: support multiple cal matrix input types
parser.add_argument("--cal-dir", default="camera_cal/", help="Relative path to the directory containing camera calibration form KITTI. \
                    Default is camera_cal/")

parser.add_argument("--video", action="store_true", help="Weather or not to advance frame-by-frame as fast as possible. \
                    By default, this will pull images from ./eval/video")

parser.add_argument("--show-yolo", action="store_true", help="Show the 2D BoundingBox detecions on a separate image")

parser.add_argument("--hide-debug", action="store_true", help="Supress the printing of each 3d location")


def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, cam_to_img, orient, dimensions, location)  # 3d boxes

    return location


def main():

    # 默认值：cal_dir='camera_cal/', hide_debug=False, image_dir='eval/image_2/', show_yolo=False, video=False
    FLAGS = parser.parse_args()

    # 注意：总共有两个权重文件，一个是yolo2D检测的yolov3.weights权重文件
    # 一个是自己训练的回归维度和alpha的权重文件，命名为epoch_10.pkl
    weights_path = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'weights' + os.path.sep
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s' % model_lst[-1])

        # 采用vgg19_bn来提取图片的特征，该特征作为后面3个branch的输入特征
        # TODO 是否要换成VGG16_bn?
        my_vgg = vgg.vgg19_bn(pretrained=True)

        # TODO: load bins from file or something
        model = Model.Model(features=my_vgg.features, bins=2)

        # 在CPU上进行测试
        checkpoint = torch.load(weights_path + '/%s' % model_lst[-1], map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # load yolo
    yolo_path = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'weights' + os.path.sep
    yolo = cv_Yolo(yolo_path)

    # 训练集中统计的各个class的维度统计信息
    averages = ClassAverages.ClassAverages()

    # TODO: clean up how this is done. flag?
    angle_bins = generate_bins(2)

    # 待检测图片的途径
    image_dir = FLAGS.image_dir

    # 当所有的图片用的是同一个proj_matrix时，应该将该proj_matrix放在该目录下
    cal_dir = FLAGS.cal_dir

    # FLAGS.video默认为false
    if FLAGS.video:
        if FLAGS.image_dir == "eval/image_2/" and FLAGS.cal_dir == "camera_cal/":
            image_dir = "eval/video/2011_09_26/image_2/"
            cal_dir = "eval/video/2011_09_26/"

    img_path = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + image_dir
    # using P_rect from global calibration file
    # calib_path = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + cal_dir
    # calib_file = calib_path + "calib_cam_to_cam.txt"

    # using P from each frame
    calib_path = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'eval' + os.path.sep + 'calib' + os.path.sep

    try:
        ids = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
    except:
        print("\nError: no images in %s" % img_path)
        exit()

    for img_id in ids:

        start_time = time.time()

        img_file = img_path + img_id + ".png"

        # P for each frame
        calib_file = calib_path + img_id + ".txt"

        truth_img = cv2.imread(img_file)
        img = np.copy(truth_img)
        yolo_img = np.copy(truth_img)

        # yolo检测出来的结果为2d像素坐标和类别
        detections = yolo.detect(yolo_img)

        for detection in detections:

            # 检测的类别必须出现在KITTI数据集的枚举的类别中，如果不在，那么忽视这个被检测出来的类别
            # 因为yolo定义的类别数量是比KITTI数据集的类别数量多，所以可能yolo检测出了一个类别，但没有出现
            # 在KITTI数据集的枚举类别中
            if not averages.recognized_class(detection.detected_class):
                continue

            # this is throwing when the 2d bbox is invalid
            # TODO: better check
            # 将图像 以及检测到的类别，2D框 以及对应这张图像的proj_matrix作为参数传入到DetectedObject类的init()函数中
            try:
                detectedObject = DetectedObject(img, detection.detected_class, detection.box_2d, calib_file)
            except:
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = detection.box_2d
            detected_class = detection.detected_class

            input_tensor = torch.zeros([1, 3, 224, 224])
            input_tensor[0, :, :, :] = input_img

            # 得到预测的orient,conf,dim
            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]
            dim += averages.get_item(detected_class)
            # 取conf大的那个bin，将该bin对应的orient的值赋值给最终的orient
            argmax = np.argmax(conf)
            orient = orient[argmax, :]

            # 得到预测出来的cos值和sin值
            # cos值在训练集中是cos(angle_diff)，sin值在训练集中是sin(angle_diff)
            # 而angle_diff是真实的alpha(经过扩展到0-2pi)与对应的bin的夹角
            cos = orient[0]
            sin = orient[1]

            # np.arctan2传入sin为y轴坐标
            # cos为x轴坐标
            # 返回弧度制角度 -pi~+pi
            # 参考https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.arctan2.html
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi  # 得到最终的alpha的值

            # 展示2D检测效果，默认不展示
            if FLAGS.show_yolo:
                location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)
            else:
                location = plot_regressed_3d_bbox(img, proj_matrix, box_2d, dim, alpha, theta_ray)

            if not FLAGS.hide_debug:  # FLAGS.hide_debug默认为False

                # 对于每一个检测到的类输出其位置信息。为了保证与KITTI数据集中的一致
                # 进行 location[1] += dim[0]
                location[1] += dim[0] / 2
                print('Estimated pose: %s' % location)

        if FLAGS.show_yolo:  # FLAGS.show_yolo默认为False
            numpy_vertical = np.concatenate((truth_img, img), axis=0)
            cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)
        else:
            cv2.imshow('3D detections', img)

        if not FLAGS.hide_debug:
            print('Got %s poses in %.3f seconds' % (len(detections), time.time() - start_time))
            print('-------------')

        if FLAGS.video:
            cv2.waitKey(1)
        else:
            if cv2.waitKey(0) != 32:  # space bar
                exit()


if __name__ == '__main__':
    main()
