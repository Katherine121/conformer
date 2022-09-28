import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_data(datapath="process_dis", path_len=15, seq_len=10):
    all_path = []
    all_labels = []
    all_pos = []

    file1 = open(datapath + "/" + "all_path.txt", 'r')
    for path in file1:
        path = path.strip('\n')  # 将\n去掉
        path = path.split(' ')[:path_len]  # 按照逗号分开

        # 图片路径
        all_path.append(path[:seq_len])
        all_path.append(path[-1:-(seq_len+1):-1])
    file1.close()

    file1 = open(datapath + "/" + "all_label.txt", 'r')
    for pos in file1:
        pos = pos.strip('\n')  # 将\n去掉
        pos = pos.split(' ')[:2*path_len]  # 按照逗号分开
        # 注意这里是5个0
        pos = [float(pos[i]) * 10000 for i in range(0, 2*path_len)]
        pos = [pos[i:i + 2] for i in range(0, len(pos), 2)]  # 两个一组

        first = []
        for i in range(1, seq_len+1):
            first.append((pos[i][0] - pos[i - 1][0], pos[i][1] - pos[i - 1][1]))
        second = []
        for i in range(-2, -(seq_len+2), -1):
            second.append((pos[i][0] - pos[i + 1][0], pos[i][1] - pos[i + 1][1]))

        # 预测下一张图片的位移
        all_labels.append(first)
        all_labels.append(second)

        # 当前图片的位置
        all_pos.append(pos[:seq_len])
        all_pos.append(pos[-1:-(seq_len+1):-1])
    file1.close()

    return all_path, all_labels, all_pos


class TrainDataset(Dataset):
    def __init__(self, transform, datapath="process_dis", path_len=15, seq_len=10):
        self.transform = transform
        res = []
        all_path, all_labels, all_pos = get_data(datapath=datapath, path_len=path_len, seq_len=seq_len)
        for i in range(0, len(all_path)):
            if i % 5 != 0:
                res.append((all_path[i], all_labels[i], all_pos[i]))

        print(len(res))
        self.imgs = res

    # 返回数据集大小
    def __len__(self):
        return len(self.imgs)

    # 打开index对应图片进行预处理后return回处理后的图片和标签
    def __getitem__(self, index):
        path, label, pos = self.imgs[index]
        # seq_len, c, h, w
        pics = None
        for pic in path:
            # 将../转换成./
            pic = Image.open(pic[1:])
            pic = pic.convert('RGB')
            pic = self.transform(pic)

            if pics is None:
                pics = pic.flatten().unsqueeze(dim=0)
            else:
                pics = torch.cat((pics, pic.flatten().unsqueeze(dim=0)), dim=0)

        # seq_len, 2
        label = torch.tensor(label, dtype=torch.float64)
        pos = torch.tensor(pos, dtype=torch.float64)

        return pics, label, pos


class TestDataset(Dataset):
    def __init__(self, transform, datapath="process_dis", path_len=15, seq_len=10):
        self.transform = transform
        res = []
        all_path, all_labels, all_pos = get_data(datapath=datapath, path_len=path_len, seq_len=seq_len)
        for i in range(0, len(all_path)):
            if i % 5 == 0:
                res.append((all_path[i], all_labels[i], all_pos[i]))

        print(len(res))
        self.imgs = res

    # 返回数据集大小
    def __len__(self):
        return len(self.imgs)

    # 打开index对应图片进行预处理后return回处理后的图片和标签
    def __getitem__(self, index):
        path, label, pos = self.imgs[index]
        # seq_len, c, h, w
        pics = None
        for pic in path:
            # 将../转换成./
            pic = Image.open(pic[1:])
            pic = pic.convert('RGB')
            pic = self.transform(pic)

            if pics is None:
                pics = pic.flatten().unsqueeze(dim=0)
            else:
                pics = torch.cat((pics, pic.flatten().unsqueeze(dim=0)), dim=0)

        # seq_len, 2
        label = torch.tensor(label, dtype=torch.float64)
        pos = torch.tensor(pos, dtype=torch.float64)

        return pics, label, pos


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 1：31143+25598+9981 *2*0.8
    trainDataset = TrainDataset(transform=transform, datapath=".", path_len=15, seq_len=10)
    trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=False, drop_last=False)
    # 1：31143+25598+9981 *2*0.2
    testDataset = TestDataset(transform=transform, datapath=".", path_len=15, seq_len=10)
