import math
import os

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

import autoaugment
from conformer import Conformer
from process_dis.dis_dataloader import TrainDataset, TestDataset


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

            # 每800个iteration打印一次测试集准确率
            if t > 0 and t % 800 == 0:
                print('下一个位置的经纬度的米的误差' + str(diff.sum() / b / seq_len / 2))
                print('终点的经纬度的米的误差' + str(diff[:, -1, :].sum() / b / 2))
            t += 1

        diff1 = float(total_diff) / total_samples
        diff2 = float(end_diff) / end_samples

    return diff1, diff2


def train(
        loader_train=None,
        loader_val=None,
        device=None,
        model=None,
        criterion=nn.CrossEntropyLoss(),
        scheduler=None,
        optimizer=None,
        epochs=300,
        check_point_dir=None
):
    diff1 = 0
    diff2 = 0
    losses = []
    best_diff = math.inf

    for e in range(epochs):
        model.train()
        total_loss = 0
        for t, (x, y, _) in enumerate(loader_train):
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            outputs = model(x)

            # 原y+混y和原t，混t求损失：lam越大，小方块越小，被识别成真图片的概率越大
            # 2
            loss = criterion(outputs, y)
            loss_value = np.array(loss.item())
            total_loss += loss_value

            # 1
            optimizer.zero_grad()
            # 3
            loss.backward()
            # 4
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # 800个iteration打印一下训练集损失
            if t > 0 and t % 800 == 0:
                print("Iteration:" + str(t) + ', average Loss = ' + str(loss_value))

        total_loss /= t
        losses.append(total_loss)

        diff1, diff2 = check_accuracy(loader_val, model, device=device)

        # 每个epoch记录一次测试集准确率和所有batch的平均训练损失
        print("Epoch:" + str(e) +
              ', Val diff1 = ' + str(diff1) +
              ', Val diff2 = ' + str(diff2) +
              ', average Loss = ' + str(total_loss))

        if os.path.exists(check_point_dir) is False:
            os.mkdir(check_point_dir)

        # 将每个epoch的平均损失写入文件
        with open(check_point_dir + "/" + "avgloss.txt", "a") as file1:
            file1.write(str(total_loss) + '\n')
        file1.close()
        # 将每个epoch的测试集准确率写入文件
        with open(check_point_dir + "/" + "testdiff.txt", "a") as file2:
            file2.write(str(diff1) + ' ' + str(diff2) + '\n')
        file2.close()

        # 如果到了保存的epoch或者是训练完成的最后一个epoch
        if diff2 < best_diff:
            best_diff = diff2
            model.eval()
            # 保存模型参数
            torch.save(model.state_dict(), check_point_dir + "/" + "model.pth")
            # 保存模型结构
            torch.save(model, check_point_dir + "/" + "model.pt")

    return diff1, diff2


if __name__ == '__main__':
    print('############################### Dataset loading ###############################')

    # 记得修改check_accuracy / 100000
    path_len = 15
    seq_len = 10
    datapath = "process_dis"
    check_point_dir = "saved_model2"

    transform = transforms.Compose([
        transforms.Resize((160, 90)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # transform_aug = transforms.Compose([
    #     transforms.Resize((160, 90)),
    #     autoaugment.CIFAR10Policy(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])

    # 原图
    trainDataset = TrainDataset(transform=transform, datapath=datapath,
                                path_len=path_len, seq_len=seq_len)
    # # 数据增强
    # trainDataset_aug = TrainDataset(transform=transform_aug, datapath=datapath,
    #                                 path_len=path_len, seq_len=seq_len)

    trainLoader = DataLoader(trainDataset,
                             batch_size=32, shuffle=True, drop_last=False)

    # 原图
    testDataset = TestDataset(transform=transform, datapath=datapath,
                              path_len=path_len, seq_len=seq_len)

    testLoader = DataLoader(testDataset,
                            batch_size=16, shuffle=True, drop_last=False)

    print('###############################  Dataset loaded  ##############################')

    print('############################### Model loading ###############################')

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device('cuda')

    # 1加载模型结构
    # 2加载模型权重
    model = Conformer(num_classes=seq_len,
                      input_dim=3 * 160 * 90,
                      encoder_dim=32,
                      num_encoder_layers=3)
    # model = torch.load(check_point_dir + "/model.pt")

    # 3设置运行环境
    model = model.to(device)

    print('###############################  Model loaded  ##############################')

    lr = 0.0001
    wd = 0.3
    epochs = 300
    save_epochs = 3

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2)

    args = {
        'loader_train': trainLoader,
        'loader_val': testLoader,
        'device': device,
        'model': model,
        'criterion': nn.MSELoss(),
        'scheduler': lr_scheduler,
        'optimizer': optimizer,
        'epochs': epochs,
        'check_point_dir': check_point_dir
    }
    train(**args)
    # diff = check_accuracy(testLoader, model, device)
