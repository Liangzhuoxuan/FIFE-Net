import cv2
import numpy as np
import os
import random


def calculate_white_pixel_pro(path)->float:
    """计算二值化图像的白色像素点面积"""
    img = cv2.imread(path)
    h = img.shape[0]
    w = img.shape[1]

    black = 0
    white = 0

    for i in range(h):
        for j in range(w):
            value = img[i][j][0]  # binary之后的图像只有一个通道
            if value == 0:
                black += 1
            else:
                white += 1

    assert white+black == 512*512

    white_pro = white/(black+white)
    # print(white, black, white/(black+white))
    print(white_pro)
    return white_pro

def preprocess_dataset_files()->None:
    """
    train: A、B、label
    test: A、B、label
    val: A、B、label
    """

    origin_path = "./67HR/67deleteblackHR/"
    folder_name = ["val", "train", "test"]
    three = ['A', 'B', 'label']

    for filename in filename_list:
        for folder in folder_name:
            current_path = origin_path + folder
            for _ in three:
                # 组成完整的路径字符串如 67HR/67deleteblackHR/val/A
                current_path = current_path + '/' + _

                f = os.walk(current_path)
                filename_list = list()  # 目录下所有文件名
                
                for _, __, filename in f:
                    filename_list = filename
                
                # 从文件名里面挑出 100 个来调试程序
                filename_list = filename_list[:100]

def delete_three(num_to_delete, folder:str, threshold=0.03)->None:
    """
    folder: train / test / val

    求解方程: (5020-y)/(8372-2x-y)=0.6 得 y=3x
              8372 - 2x - y = 4300
              得 x=800, y=2400
    """
    folder_a_path = "E:/MobaXterm_Download/67HR/67deleteblackHR/"+ folder +"/A"
    folder_b_path = "E:/MobaXterm_Download/67HR/67deleteblackHR/"+ folder +"/B"
    folder_c_path = "E:/MobaXterm_Download/67HR/67deleteblackHR/"+ folder +"/label"

    file_names = set(os.listdir(folder_a_path))

    files_to_delete = list()

    for file_name in file_names:
        white_proportion = calculate_white_pixel_pro(folder_c_path + '/' + file_name)
        if white_proportion < threshold:
            files_to_delete.append(file_name)

    # files_to_delete = random.sample(file_names, num_to_delete)

    for folder_path in [folder_a_path, folder_b_path, folder_c_path]:
        for file_name in files_to_delete:
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
                print("delete", file_path)



if __name__ == "__main__":
    path = "./67HR/67deleteblackHR/val/label/0305-6780-LA93_6144_0_6656_512.tif"
    # calculate_white_pixel_pro(path)

    threshold = 0.03  # 白色像素点面积的阈值

    # preprocess_dataset_files()
    delete_three(200, "train", threshold=threshold)
    delete_three(200, "test", threshold=threshold)
    delete_three(200, "val", threshold=threshold)

