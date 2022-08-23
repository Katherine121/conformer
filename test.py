import math
import os
import time

import torch
import torch.nn as nn
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms import autoaugment

from conformer import Conformer
from process_dis.dis_dataloader import TrainDataset, TestDataset
# from process_dis2.dis_dataloader import TrainDataset, TestDataset


def check_accuracy(loader, model, device=None):
    model.eval()
    total_diff = 0
    total_samples = 0

    with torch.no_grad():
        t = 0
        for x, y, pos in loader:
            x = x.to(device, dtype=torch.float32)
            # 理论位移
            y = y.to(device, dtype=torch.float64)
            # 理论起点
            pos = pos.to(device, dtype=torch.float64)

            # 输出位移
            outputs = model(x)
            outputs = outputs.to(dtype=torch.float64)

            b, seq_len, _ = x.size()
            for i in range(0, seq_len):
                # 理论起点+理论位移=理论下一个位置
                y[:, i, :] = pos[:, i, :] + y[:, i, :]
                # 理论起点+输出位移=输出下一个位置
                outputs[:, i, :] = pos[:, i, :] + outputs[:, i, :]

            # 理论下一个位置的经纬度
            lat = y / 100000
            # lat = y / 10000

            # 输出与理论下一个位置的经纬度误差
            diff = torch.abs(outputs - y)

            # 输出与理论下一个位置的纬度误差的米
            diff *= 1.11
            # diff *= 11.1
            # 输出与理论下一个位置的经度误差的米
            diff[:, :, 1] *= torch.cos(lat[:, :, 0] * math.pi / 180)

            # 误差总和
            diff = diff.sum()
            total_diff += diff
            total_samples += b * seq_len * 2

            if t % 10 == 0:
                print(t)
            t += 1

        diff = float(total_diff) / total_samples

    print(diff)

    return diff


if __name__ == '__main__':
    print('############################### Dataset loading ###############################')

    # 记得修改check_accuracy / 100000
    path_len = 15
    seq_len = 10
    datapath = "process_dis"
    check_point_dir = "saved_model"

    transform = transforms.Compose([
        transforms.Resize((160, 90)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # 原图
    testDataset = TestDataset(transform=transform, datapath=datapath, pic_path="whole_path",
                              path_len=path_len, seq_len=seq_len)

    testLoader = DataLoader(testDataset,
                            batch_size=16, shuffle=True, drop_last=False)

    print('###############################  Dataset loaded  ##############################')

    print('############################### Model loading ###############################')

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda')

    # 1加载模型结构
    # 2加载模型权重
    model = torch.load(check_point_dir + "/model.pt")

    # 3设置运行环境
    model = model.to(device)

    print('###############################  Model loaded  ##############################')

    diff = check_accuracy(testLoader, model, device)
