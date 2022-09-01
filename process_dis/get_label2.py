import operator
import os
import shutil
import random

import numpy as np
from PIL import Image


def sort_pics(path):
    pics_list = []
    labels = []

    file_path = os.listdir(path)
    file_path = sorted(file_path)
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

        labels.append(list(map(eval, [lat_pos, alt_pos, lon_pos, start])))

    return pics_list, labels


def gene_path(path, path_len):
    all_path = []
    all_label = []

    dir_path = os.listdir(path)
    dir_path = sorted(dir_path)
    for dir in dir_path:

        full_dir_path = os.path.join(path, dir)
        # 一个文件夹内的完整路径
        pics_list, labels = sort_pics(full_dir_path)

        pics_len = len(pics_list)

        for i in range(0, pics_len - path_len + 1):
            # 一整条路径
            item = []
            lab = []
            # 加入起点
            item.append(pics_list[i])
            lab.append(labels[i])
            # 上一个点
            last_pos = labels[i]

            # 加入后续点
            for j in range(i + 1, pics_len):

                if abs(labels[j][0] - last_pos[0]) < 10e-5 and abs(labels[j][2] - last_pos[2]) < 10e-5:
                    continue
                if abs(labels[j][0] - last_pos[0]) > 50e-5 or abs(labels[j][2] - last_pos[2]) > 50e-5:
                    continue
                # if abs(labels[j][1] - last_pos[1]) > 3:
                #     continue

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
    path = "../order"

    # 原数据集的方向
    all_path, all_label = gene_path(path, path_len=15)

    # 10+50:9981
    # 10+50+不限高:22133
    # write_path(all_path, all_label)

    # i = 30
    # for pic in all_path[10000]:
    #     img = Image.open(pic)
    #     img.save("example" + "/" + str(i) + ".png")
    #     i += 1
    # for pic in all_path[20000]:
    #     img = Image.open(pic)
    #     img.save("example" + "/" + str(i) + ".png")
    #     i += 1
