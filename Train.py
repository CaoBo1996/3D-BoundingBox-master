from torch_lib.Dataset import *
from torch_lib.Model import Model, OrientationLoss

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg
from torch.utils import data

import os

# 训练的时候，不需要相机校正参数，但是DetectedObject类在测试集上也用到了
# 而测试集中需要用要相机参数，所以为了代码的兼容性，所以，训练集会加载
# 一个global cal_matrix，但是并不起什么作用


def main():

    # hyper parameters
    epochs = 100
    batch_size = 8  # 批训练数据的个数
    alpha = 0.6
    w = 0.4

    print("Loading all detected objects in dataset...")

    # 找到训练集的路径，目录默认为 ./Kitti/training/
    train_path = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'Kitti' + os.path.sep + 'training' + os.path.sep

    # 执行Dataset()的init函数
    dataset = Dataset(train_path)

    # shuffle为true表示打乱数据 ，num_works线程个数
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 6}

    generator = data.DataLoader(dataset, **params)

    my_vgg = vgg.vgg19_bn(pretrained=True)
    model = Model(features=my_vgg.features)
    opt_SGD = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    conf_loss_func = nn.CrossEntropyLoss()
    dim_loss_func = nn.MSELoss()

    # 对于orient的损失函数，采用自定义的损失函数
    orient_loss_func = OrientationLoss

    # load any previous weights
    model_path = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'weights' + os.path.sep
    latest_model = None
    first_epoch = 0
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    else:
        try:
            latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
        except:
            pass

    if latest_model is not None:
        checkpoint = torch.load(model_path + latest_model, map_location=torch.device('cpu'))  # 加载epoch_10.pkl文件
        model.load_state_dict(checkpoint['model_state_dict'])
        opt_SGD.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print('Found previous checkpoint: %s at epoch %s' % (latest_model, first_epoch))
        print('Resuming training....')

    total_num_batches = int(len(dataset) / batch_size)

    for epoch in range(first_epoch + 1, epochs + 1):
        curr_batch = 0
        passes = 0
        for local_batch, local_labels in generator:

            # Orientation是根据angle角与bin的中心角度的差计算的cos和sin的值
            # 注意此处的bin是angle落在哪个bin中，没落的bin对应者的orient为0，0
            truth_orient = local_labels['Orientation'].float()

            # 根据label中angle落在哪个bin上，得到的confidence信息，由于本文设置的bin
            # 的个数为2，所以对于每一个label标签中的每一行，Confidence都是1*2矩阵
            truth_conf = local_labels['Confidence'].long()

            # 标签中的真正的维度信息
            truth_dim = local_labels['Dimensions'].float()

            local_batch = local_batch.float()

            # 数据送入到模型中，得到预测的结果
            [orient, conf, dim] = model(local_batch)

            orient_loss = orient_loss_func(orient, truth_orient, truth_conf)
            dim_loss = dim_loss_func(dim, truth_dim)

            # 返回的是truth_conf为1的的索引下标
            truth_conf = torch.max(truth_conf, dim=1)[1]
            conf_loss = conf_loss_func(conf, truth_conf)

            loss_theta = conf_loss + w * orient_loss
            loss = alpha * dim_loss + loss_theta

            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()

            if passes % 10 == 0:
                print("--- epoch %s | batch %s/%s --- [loss: %s]" % (epoch, curr_batch, total_num_batches, loss.item()))
                passes = 0

            passes += 1
            curr_batch += 1

        # save after every 10 epochs
        if epoch % 10 == 0:
            name = model_path + 'epoch_%s.pkl' % epoch
            print("====================")
            print("Done with epoch %s!" % epoch)
            print("Saving weights as %s ..." % name)
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt_SGD.state_dict(), 'loss': loss}, name)
            print("====================")


if __name__ == '__main__':
    main()
