import math
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from cluster.class_dataloader import TrainDataset, TestDataset, get_center
from conformer.classify_model import ClassifyConformer
from utils import autoaugment


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

            t += 1

        acc = float(total_correct) / total_samples
        diff = float(total_diff) / total_samples
    return acc, diff


def train(
        loader_train=None,
        loader_val=None,
        centers=None,
        device=None,
        model=None,
        criterion=nn.CrossEntropyLoss(),
        scheduler=None,
        optimizer=None,
        epochs=300,
        check_point_dir=None
):
    acc = 0
    diff = 0
    best_acc = 0

    for e in range(epochs):
        model.train()
        total_loss = 0
        for t, (x, y, _) in enumerate(loader_train):
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)

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

            # 400个iteration打印一下训练集损失
            if t > 0 and t % 400 == 0:
                print("Iteration:" + str(t) + ', average Loss = ' + str(loss_value))

        total_loss /= t

        acc, diff = check_accuracy(loader_val, model, device=device, centers=centers)

        # 每个epoch记录一次测试集准确率和所有batch的平均训练损失
        print("Epoch:" + str(e) +
              ', Val acc = ' + str(acc) +
              ', Val diff = ' + str(diff) +
              ', average Loss = ' + str(total_loss))

        if os.path.exists(check_point_dir) is False:
            os.mkdir(check_point_dir)

        # 将每个epoch的平均损失写入文件
        with open(check_point_dir + "/" + "avgloss.txt", "a") as file1:
            file1.write(str(total_loss) + '\n')
        file1.close()
        # 将每个epoch的测试集准确率写入文件
        with open(check_point_dir + "/" + "testacc.txt", "a") as file2:
            file2.write(str(acc) + ' ' + str(diff) + '\n')
        file2.close()

        # 如果到了保存的epoch或者是训练完成的最后一个epoch
        if acc > best_acc:
            best_acc = acc
            model.eval()
            # 保存模型参数
            torch.save(model.state_dict(), check_point_dir + "/" + "model.pth")
            # 保存模型结构
            torch.save(model, check_point_dir + "/" + "model.pt")

    return acc, diff


if __name__ == '__main__':
    print('############################### Dataset loading ###############################')

    datapath = "cluster"
    check_point_dir = "saved_model2"
    class_num = 150
    max_num = 400
    centers = get_center(path=datapath)

    transform = transforms.Compose([
        transforms.Resize((90, 160)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_crop = transforms.Compose([
        transforms.CenterCrop((90, 160)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_cifar = transforms.Compose([
        transforms.Resize((90, 160)),
        autoaugment.CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_erase = transforms.Compose([
        transforms.Resize((90, 160)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing(),
    ])

    # 原图
    trainDataset = TrainDataset(transform=transform, datapath=datapath, class_num=class_num, max_num=max_num)
    # 数据增强
    trainDataset_crop = TrainDataset(transform=transform_crop, datapath=datapath, class_num=class_num, max_num=max_num)
    trainDataset_cifar = TrainDataset(transform=transform_cifar, datapath=datapath, class_num=class_num, max_num=max_num)
    trainDataset_erase = TrainDataset(transform=transform_erase, datapath=datapath, class_num=class_num, max_num=max_num)

    trainLoader = DataLoader(trainDataset + trainDataset_crop + trainDataset_cifar + trainDataset_erase,
                             batch_size=64, shuffle=True, drop_last=False)

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
    seq_model = torch.load("saved_model/model.pt")
    # 加载分类模型
    model = ClassifyConformer(num_classes=class_num,
                              input_dim=3 * 30 * 40,
                              encoder_dim=32,
                              num_encoder_layers=3)

    # 读取预训练序列模型参数
    pretrained_dict = seq_model.state_dict()
    print(len(pretrained_dict))
    # 读取分类模型参数
    net_dict = model.state_dict()
    print(len(net_dict))

    # 将pretrained_dict里不属于net_dict的键剔除掉
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
    # 更新修改之后的net_dict
    net_dict.update(pretrained_dict)
    print(len(pretrained_dict))
    # 加载我们真正需要的state_dict
    model.load_state_dict(net_dict)

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
        'centers': centers,
        'device': device,
        'model': model,
        'criterion': nn.CrossEntropyLoss(),
        'scheduler': lr_scheduler,
        'optimizer': optimizer,
        'epochs': epochs,
        'check_point_dir': check_point_dir
    }
    train(**args)
    # acc, diff = check_accuracy(testLoader, model, device, centers)
