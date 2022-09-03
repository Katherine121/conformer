import math
import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from process_dis.dis_dataloader import TestDataset


def check_accuracy(loader, model, device=None):
    model.eval()
    total_diff = 0
    total_samples = 0
    end_diff = 0
    end_samples = 0

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

            y[:, 0, :] += pos[:, 0, :]
            outputs[:, 0, :] += pos[:, 0, :]
            for i in range(1, seq_len):
                # 理论起点+理论位移=理论下一个位置
                y[:, i, :] += y[:, i - 1, :]
                # 理论起点+输出位移=输出下一个位置
                outputs[:, i, :] += outputs[:, i - 1, :]

            # 理论下一个位置的经纬度
            lat = y / 10000

            # 输出与理论下一个位置的经纬度误差
            diff = torch.abs(outputs - y)

            # 输出与理论下一个位置的纬度误差的米
            diff *= 11.1
            # 输出与理论下一个位置的经度误差的米
            diff[:, :, 1] *= torch.cos(lat[:, :, 0] * math.pi / 180)

            # 误差总和
            total_diff += diff.sum()
            total_samples += b * seq_len * 2

            end_diff += diff[:, -1, :].sum()
            end_samples += b * 2

            t += 1
            if t % 20 == 0:
                print(t)

        diff1 = float(total_diff) / total_samples
        diff2 = float(end_diff) / end_samples

        print(diff1)
        print(diff2)
    return diff1, diff2


if __name__ == '__main__':
    print('############################### Dataset loading ###############################')

    # 记得修改check_accuracy / 100000
    path_len = 15
    seq_len = 10
    datapath = "process_dis"
    check_point_dir = "saved_model"

    transform = transforms.Compose([
        transforms.Resize((90, 160)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # 原图
    testDataset = TestDataset(transform=transform, datapath=datapath,
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
