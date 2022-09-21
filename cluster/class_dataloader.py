import shutil

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
import os

from torchvision import transforms

torch.set_printoptions(profile="full")
torch.set_printoptions(precision=15)


# 获取所有类别的簇中心坐标
def get_center(path="cluster"):
    center = None

    f = open(os.path.join(path, "cluster_centre.txt"), 'rt')
    for line in f:
        line = line.replace('\n', '')
        cur = torch.tensor(list(map(eval, line.split(' '))), dtype=torch.float64).unsqueeze(dim=0)
        if center is None:
            center = cur
        else:
            center = torch.cat((center, cur), dim=0)
    f.close()

    return center


def get_train_data(datapath="cluster", max_num=400):
    # 最大数目1023*0.8，最小数目42*0.8
    all_pics = []
    f = open(os.path.join(datapath, "cluster_pics.txt"), 'rt')
    for line in f:
        line = line.replace('\n', '')
        line = line.split(' ')

        pics = []
        for i in range(0, len(line)):
            if i % 5 != 0:
                pics.append(line[i])

        if len(pics) < max_num:
            for j in range(0, max_num - len(pics)):
                pics.append(pics[j % len(pics)])

        all_pics.append(pics)
    f.close()

    all_labels = []
    f = open(os.path.join(datapath, "cluster_labels.txt"), 'rt')
    for line in f:
        line = line.replace('\n', '')
        line = line.split(' ')
        line = [float(x) for x in line]
        line = [line[i:i + 2] for i in range(0, len(line), 2)]  # 两个一组

        labels = []
        for i in range(0, len(line)):
            if i % 5 != 0:
                labels.append(line[i])

        if len(labels) < max_num:
            for j in range(0, max_num - len(labels)):
                labels.append(labels[j % len(labels)])

        all_labels.append(labels)
    f.close()

    return all_pics, all_labels


def get_test_data(datapath="cluster"):
    # 最大数目1023*0.2，最小数目42*0.2
    all_pics = []
    f = open(os.path.join(datapath, "cluster_pics.txt"), 'rt')
    for line in f:
        line = line.replace('\n', '')
        line = line.split(' ')

        pics = []
        for i in range(0, len(line)):
            if i % 5 == 0:
                pics.append(line[i])

        all_pics.append(pics)
    f.close()

    all_labels = []
    f = open(os.path.join(datapath, "cluster_labels.txt"), 'rt')
    for line in f:
        line = line.replace('\n', '')
        line = line.split(' ')
        line = [float(x) for x in line]
        line = [line[i:i + 2] for i in range(0, len(line), 2)]  # 两个一组

        labels = []
        for i in range(0, len(line)):
            if i % 5 == 0:
                labels.append(line[i])

        all_labels.append(labels)
    f.close()

    return all_pics, all_labels


class TrainDataset(Dataset):
    def __init__(self, transform, datapath="cluster", class_num=100, max_num=400):
        self.transform = transform
        res = []
        all_pics, all_labels = get_train_data(datapath, max_num)
        for i in range(0, class_num):
            for j in range(0, len(all_pics[i])):
                res.append((all_pics[i][j], i, all_labels[i][j]))

        print(len(res))
        self.imgs = res

    # 返回数据集大小
    def __len__(self):
        return len(self.imgs)

    # 打开index对应图片进行预处理后return回处理后的图片和标签
    def __getitem__(self, index):
        pic, label, pos = self.imgs[index]

        pic = Image.open(pic[1:])
        pic = pic.convert('RGB')
        pic = self.transform(pic)

        pos = torch.tensor(pos, dtype=torch.float64)

        return pic, label, pos


class TestDataset(Dataset):
    def __init__(self, transform, datapath="cluster", class_num=100):
        self.transform = transform
        res = []
        all_pics, all_labels = get_test_data(datapath)
        for i in range(0, class_num):
            for j in range(0, len(all_pics[i])):
                res.append((all_pics[i][j], i, all_labels[i][j]))

        print(len(res))
        self.imgs = res

    # 返回数据集大小
    def __len__(self):
        return len(self.imgs)

    # 打开index对应图片进行预处理后return回处理后的图片和标签
    def __getitem__(self, index):
        pic, label, pos = self.imgs[index]

        pic = Image.open(pic[1:])
        pic = pic.convert('RGB')
        pic = self.transform(pic)

        pos = torch.tensor(pos, dtype=torch.float64)

        return pic, label, pos


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((90, 160)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    trainDataset = TrainDataset(transform=transform, datapath=".", class_num=150, max_num=400)
    testDataset = TestDataset(transform=transform, datapath=".", class_num=150)

    # f = open("cluster_pics.txt", 'rt')
    # for line in f:
    #     line = line.replace('\n', '')
    #     line = line.split(' ')
    #     pic = line[0]
    #
    #     pic = Image.open(pic)
    #     pic = pic.convert('RGB')
    #     pic = transform(pic).flatten()
    #     print(pic)
    #
    #     break
    # f.close()
