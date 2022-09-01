import operator
import os
import shutil
import random

import numpy as np
from PIL import Image


def remove_pics(path1, path2):
    if os.path.exists(path2) is False:
        os.mkdir(path2)

    dir_path = os.listdir(path1)
    for dir in dir_path:

        full_dir_path = os.path.join(path1, dir)

        file_path = os.listdir(full_dir_path)
        for file in file_path:

            full_file_path = os.path.join(full_dir_path, file)

            # 删除没有坐标的图片
            if 'None' in full_file_path:
                os.remove(full_file_path)
                continue

            # 删除打不开的图片
            try:
                image = Image.open(full_file_path)
                image = image.convert('RGB')
            except (OSError, NameError):
                os.remove(full_file_path)
                continue

            shutil.copy(full_file_path, path2)


def sort_pics(path1, lat_or_lon=0):
    # process pics
    pics_list = []
    labels = []

    file_path = os.listdir(path1)

    for file in file_path:
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

        labels.append(list(map(eval, [lat_pos, alt_pos, lon_pos, start])))

    if lat_or_lon is 0:
        # 0按照先纬度后经度从小到大排序
        labels.sort(key=operator.itemgetter(0, 2, 1), reverse=False)
    if lat_or_lon is 3:
        # 3按照先经度后纬度从小到大排序
        labels.sort(key=operator.itemgetter(2, 0, 1), reverse=False)

    # labels = sorted(labels, key=operator.itemgetter(0, 2, 1), reverse=False)

    for label in labels:
        file = str(label[3]) + "-lat-" + str(label[0]) + ",alt-" + str(label[1]) + ",lon-" + str(label[2]) + ".png"
        assert Image.open(path1 + "/" + file)
        pics_list.append(path1 + "/" + file)

    print("sort_pics: pics_list len is " + str(len(pics_list)))
    print("sort_pics: labels len is " + str(len(labels)))
    return pics_list, labels


def gene_path(pics_list, labels, path_len, flag):
    all_path = []
    all_label = []

    pics_len = len(pics_list)
    # 获得路径
    for i in range(0, pics_len - path_len + 1):
        item = []
        lab = []
        # 加入起点
        item.append(pics_list[i])
        lab.append(labels[i])
        # 上一个点
        last_pos = labels[i]

        # 纬度
        if flag == 0:
            # 加入后续点
            for j in range(i + 1, pics_len):

                if labels[j][0] - last_pos[0] < 10e-5 and abs(labels[j][2] - last_pos[2]) < 10e-5:
                    continue
                if labels[j][0] - last_pos[0] > 50e-5:
                    break
                if abs(labels[j][2] - last_pos[2]) > 50e-5:
                    continue
                if abs(labels[j][1] - last_pos[1]) > 3:
                    continue

                item.append(pics_list[j])
                lab.append(labels[j])
                last_pos = labels[j]

                if (len(item)) == path_len:
                    all_path.append(item)
                    all_label.append(lab)
                    break
        # 经度
        elif flag == 3:
            # 加入后续点
            for j in range(i + 1, pics_len):

                if labels[j][2] - last_pos[2] < 10e-5 and abs(labels[j][0] - last_pos[0]) < 10e-5:
                    continue
                if labels[j][2] - last_pos[2] > 50e-5:
                    break
                if abs(labels[j][0] - last_pos[0]) > 50e-5:
                    continue
                if abs(labels[j][1] - last_pos[1]) > 3:
                    continue

                item.append(pics_list[j])
                lab.append(labels[j])
                last_pos = labels[j]

                if (len(item)) == path_len:
                    all_path.append(item)
                    all_label.append(lab)
                    break

    print("gene_path: all_path num is " + str(len(all_path)))

    return all_path, all_label


def write_path(all_path, all_label):
    with open("./all_path.txt", "a") as file1:
        for path in all_path:
            for pic in path:
                file1.write(pic + " ")
            file1.write("\n")
    file1.close()

    with open("./all_label.txt", "a") as file1:
        for path in all_label:
            for pos in path:
                # 写入纬度和经度
                file1.write(str(pos[0]) + " " + str(pos[2]) + " ")
            file1.write("\n")
    file1.close()


if __name__ == "__main__":
    # path1 = "../order"
    path2 = "../whole_path"
    # remove_pics(path1, path2)

    # 纬度增加
    pics_list0, labels0 = sort_pics(path2, 0)
    # 经度增加
    pics_list1, labels1 = sort_pics(path2, 3)

    # 纬度
    all_path, all_label = gene_path(pics_list0, labels0, path_len=15, flag=0)

    # 经度
    all_path0, all_label0 = gene_path(pics_list1, labels1, path_len=15, flag=3)

    all_path.extend(all_path0)
    all_label.extend(all_label0)

    # 10+50:31143,25598
    # 10+50+不限高:36630,33516
    # write_path(all_path, all_label)

    # i = 0
    # for pic in all_path[15000]:
    #     img = Image.open(pic)
    #     img.save("example" + "/" + str(i) + "-" + pic[14:])
    #     i += 1
    # for pic in all_path[45000]:
    #     img = Image.open(pic)
    #     img.save("example" + "/" + str(i) + "-" + pic[14:])
    #     i += 1
