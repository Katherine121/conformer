import operator
import os
import shutil
import random

import numpy as np
from PIL import Image


def gene_path(path):
    all_path = []
    all_labels = []
    dir_path = os.listdir(path)
    dir_path = sorted(dir_path)
    for dir in dir_path:
        full_dir_path = os.path.join(path, dir)

        # 一整条路径
        one_path = []
        one_labels = []
        file_path = os.listdir(full_dir_path)
        file_path = sorted(file_path)
        for file in file_path:
            full_file_path = os.path.join(full_dir_path, file)

            # # 删除没有坐标的图片
            # if 'None' in full_file_path:
            #     os.remove(full_file_path)
            #     continue
            #
            # # 删除打不开的图片
            # try:
            #     image = Image.open(full_file_path)
            #     image = image.convert('RGB')
            # except (OSError, NameError):
            #     os.remove(full_file_path)
            #     continue

            # 因为文件夹不一样了，所以需要加入完整的文件路径
            # 本来写入的是../order，现在要改成./order
            one_path.append(full_file_path[1:])

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

            one_labels.append(list(map(eval, [lat_pos, alt_pos, lon_pos, start])))

        # 选择十一张图片，方便计算位移
        for i in range(0, len(one_path) - 10):
            # 不选择终点图片
            all_path.append(one_path[i: i+11])
            # 加入下一时刻的位移
            all_labels.append(one_labels[i: i+11])

    print("gene_path: all_path num is " + str(len(all_path)))

    return all_path, all_labels


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
                # 写入纬度和高度
                file1.write(str(pos[0]) + " " + str(pos[2]) + " ")
            file1.write("\n")
    file1.close()


if __name__ == "__main__":
    path = "../order"

    # 原数据集的方向
    all_path, all_label = gene_path(path)

    write_path(all_path, all_label)
