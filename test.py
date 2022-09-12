import math
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from conformer import Conformer
from conformer.classify_model import ClassifyConformer

from cluster.class_dataloader import TrainDataset, TestDataset, get_center
import autoaugment
from thop import profile


def compute_diff(pos_outputs, pos):
    diff = torch.abs(pos_outputs - pos)
    # 纬度差111km
    diff *= 111000
    # 经度差111km*cos纬度
    diff[:, 1] *= torch.cos(pos[:, 0] * math.pi / 180)

    diff *= diff

    diff = torch.sum(diff, dim=-1)

    diff = torch.sqrt(diff)

    diff = torch.sum(diff, dim=-1)

    return diff


def check_accuracy(loader, model, device=None, centers=None):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_diff = 0

    with torch.no_grad():
        t = 0
        for x, y, pos in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            pos = pos.to(device, dtype=torch.float64)

            outputs = model(x)

            # _,是batch_size*概率，preds是batch_size*最大概率的列号
            _, preds = outputs.max(1)

            num_correct = (preds == y).sum()
            num_samples = preds.size(0)

            total_correct += num_correct
            total_samples += num_samples

            # 获得当前batch所有图像对应类别的中心坐标
            pos_outputs = centers[preds].to(device)
            # 计算中心坐标（预测）与真实坐标（标签）的误差
            pos_diff = compute_diff(pos_outputs, pos)

            total_diff += pos_diff

            # 每100个iteration打印一次测试集准确率
            if t > 0 and t % 100 == 0:
                print('预测正确的图片数目' + str(num_correct))
                print('总共的图片数目' + str(num_samples))
                print('预测坐标与真实坐标的平均欧式距离' + str(pos_diff / num_samples))

            if t % 20 == 0:
                print(t)
            t += 1

        acc = float(total_correct) / total_samples
        diff = float(total_diff) / total_samples
    return acc, diff


if __name__ == '__main__':
    print('############################### Dataset loading ###############################')

    datapath = "cluster"
    check_point_dir = "saved_model3"
    class_num = 100
    max_num = 300
    centers = get_center(path=datapath)

    transform = transforms.Compose([
        transforms.Resize((90, 160)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 原图
    testDataset = TestDataset(transform=transform, datapath=datapath, class_num=class_num)

    testLoader = DataLoader(testDataset,
                            batch_size=32, shuffle=True, drop_last=False)

    print('###############################  Dataset loaded  ##############################')

    print('############################### Model loading ###############################')

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda')

    # 1加载预训练序列模型结构
    # 2加载预训练序列模型权重
    model = torch.load(check_point_dir + "/model.pt")

    # 3设置运行环境
    model = model.to(device)

    print('###############################  Model loaded  ##############################')

    acc, diff = check_accuracy(testLoader, model, device, centers)
    print(acc)
    print(diff)
