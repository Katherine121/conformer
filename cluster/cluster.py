import math
import random
import shutil
import numpy as np
import os


def copy_classes(datapath="cluster"):
    # 最大数目100，最小数目2
    i = 0
    f = open(os.path.join(datapath, "cluster_pics.txt"), 'rt')
    if os.path.exists(datapath + "/all_class") is False:
        os.mkdir(datapath + "/all_class")

    for line in f:
        line = line.replace('\n', '')
        line = line.split(' ')
        # print(len(line))
        class_path = datapath + "/all_class/" + str(i)
        os.mkdir(class_path)
        for j in range(0, len(line)):
            shutil.copy(line[j], class_path)

        i += 1
    f.close()


def get_pics(path="whole_path"):
    # process pics
    pics_list = []
    labels = []

    file_path = os.listdir(path)

    for file in file_path:
        full_file_path = os.path.join(path, file)

        pics_list.append(full_file_path)

        # 纬度
        lat_index = file.find("lat")
        # 高度
        alt_index = file.find("alt")
        # 经度
        lon_index = file.find("lon")

        start = file[0: lat_index - 1]
        lat_pos = file[lat_index + 4: alt_index - 1]
        alt_pos = file[alt_index + 4: lon_index - 1]
        lon_pos = file[lon_index + 4: -4]

        labels.append(list(map(eval, [lat_pos, lon_pos])))

    return pics_list, labels


# 计算欧氏距离
def euclidean_distance(pos1, pos2):
    return math.sqrt(((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2))


def images_clustering(path="whole_path", k=150, epoch=200):

    pics_list, labels = get_pics(path)

    # 随机生成k个初始聚类中心，保存为centre
    centre = np.empty((k, 2))
    for i in range(0, k):
        index = random.randint(0, len(labels) - 1)
        centre[i][0] = labels[index][0]
        centre[i][1] = labels[index][1]

    # 迭代epoch次
    for iter in range(0, epoch):
        print(iter)

        # 每个点到每个中心点的距离矩阵
        dis = np.empty((len(labels), k))
        for i in range(0, len(labels)):
            for j in range(0, k):
                dis[i][j] = euclidean_distance(labels[i], centre[j])

        # 初始化分类矩阵
        classify = []
        for i in range(0, k):
            classify.append([])

        # 比较距离并重新分成k类
        for i in range(0, len(labels)):
            List = dis[i].tolist()
            index = List.index(dis[i].min())
            # classify是从小到大添加的
            classify[index].append(i)

        # 构造新的中心点
        new_centre = np.empty((k, 2))
        for i in range(0, k):
            x_sum = 0
            y_sum = 0
            # 避免缺失簇
            if len(classify[i]) == 0:
                randindex = random.randint(0, len(labels) - 1)
                new_centre[i][0] = labels[randindex][0]
                new_centre[i][1] = labels[randindex][1]
                continue

            for j in range(0, len(classify[i])):
                x_sum += labels[classify[i][j]][0]
                y_sum += labels[classify[i][j]][1]

            new_centre[i][0] = x_sum / len(classify[i])
            new_centre[i][1] = y_sum / len(classify[i])

        # 比较新的中心点和旧的中心点是否一样
        if (new_centre == centre).all():
            break
        else:
            centre = new_centre

    for i in range(0, k):
        # 记录簇中心
        with open("./cluster_centre.txt", "a") as file1:
            file1.write(str(centre[i][0]) + " " + str(centre[i][1]) + "\n")
        file1.close()

        with open("./cluster_pics.txt", "a") as file1:
            for j in range(0, len(classify[i])):
                ori_path = pics_list[classify[i][j]]
                file1.write(ori_path)
                if j == len(classify[i]) - 1:
                    file1.write("\n")
                else:
                    file1.write(" ")
        file1.close()

        with open("./cluster_labels.txt", "a") as file1:
            for j in range(0, len(classify[i])):
                file1.write(str(labels[classify[i][j]][0]) + " " + str(labels[classify[i][j]][1]))
                if j == len(classify[i]) - 1:
                    file1.write("\n")
                else:
                    file1.write(" ")
        file1.close()

    print('迭代次数为：', iter + 1)
    print('聚类中心为：', centre)

    return centre, classify


if __name__ == "__main__":
    # images_clustering(path="../whole_path", k=150, epoch=200)
    copy_classes(".")
