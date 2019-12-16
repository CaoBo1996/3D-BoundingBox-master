import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# orient的损失函数
# 传入的是模型orient预测值，以及orient真实值以及Confidence真实值
def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):

    batch_size = orient_batch.size()[0]

    # 获取GTConf的值为1的索引下标，即
    indexes = torch.max(confGT_batch, dim=1)[1]

    # extract just the important bin
    # 抽取将真实的confidence为1的bin的oriention的值与预测出来的oritention的值
    # （注意：因为预测出来的有四个oriention，所以要取对应于Conf为1对应bin的那一对sin和con的值）
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]
    # temp1 = orientGT_batch[:, 1]  # sin
    # temp2 = orientGT_batch[:, 0]  # cos
    # 在Dataset.format_label()函数中被定义sin和cos的顺序

    # torch.atan2()函数的返回值为 -pi-pi
    # 对orientGT的sin作为atan2(x,y)函数的x，cos作为atan2(x,y)函数的y。
    theta_diff = torch.atan2(orientGT_batch[:, 1], orientGT_batch[:, 0])

    # 对估计的orient_batch计算atan2()
    estimated_theta_diff = torch.atan2(orient_batch[:, 1], orient_batch[:, 0])

    return -1 * torch.cos(theta_diff - estimated_theta_diff).mean()


class Model(nn.Module):
    def __init__(self, features=None, bins=2, w=0.4):  # feature即VGGfeature
        super(Model, self).__init__()
        self.bins = bins
        self.w = w
        self.features = features
        self.orientation = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins * 2)  # to get sin and cos
        )
        self.confidence = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins),
            # nn.Softmax()
            # nn.Sigmoid()
        )
        self.dimension = nn.Sequential(nn.Linear(512 * 7 * 7, 512), nn.ReLU(True), nn.Dropout(), nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(), nn.Linear(512, 3))

    def forward(self, x):
        x = self.features(x)  # 512 x 7 x 7
        x = x.view(-1, 512 * 7 * 7)
        orientation = self.orientation(x)  # 1 * 4向量
        orientation = orientation.view(-1, self.bins, 2)  #
        orientation = F.normalize(orientation, dim=2)
        confidence = self.confidence(x)
        dimension = self.dimension(x)
        return orientation, confidence, dimension
