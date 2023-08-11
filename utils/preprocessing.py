from PIL import Image
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import os
import shutil
import random 

I1 = Image.open('D:/SECOND_train_set/Second/val/label/03002.png')
# I1 = Image.open('E:/MobaXterm_Download/67HR/67deleteblackHR_fixed/67deleteblackHR_fixed/train/label/0305-6780-LA93_5632_1024_6144_1536.tif')
# I1 = torch.from_numpy(np.array(I1)).permute(2, 0, 1).float()
I1 = torch.from_numpy(np.array(I1))
print(I1)
# print(np.unique(I1))
# print(I1.size())


def get_binary_change_map(RGB_img, postive_label="Tree"):
    """
    将三通道图片转化为单通道图片
    postive_label \in {"water", "Tree"}
    """
    get_tree_label = 1 if postive_label == "Tree" else 0
    first_channel = RGB_img[0, ...]
    second_channel = RGB_img[1, ...]
    third_channel = RGB_img[2, ...]

    first_mask = None
    second_mask = None
    third_mask = None

    if get_tree_label:
        # 选取 亮绿色 的色素点，将其变为白色 255，剩下的变成黑色 0
        first_mask = first_channel == 0
        second_mask = second_channel == 255
        third_mask = third_channel == 0
    else:
        # 选取 蓝色 的色素点，将其变为白色 255，剩下的变成黑色 0
        first_mask = first_channel == 0
        second_mask = second_channel == 0
        third_mask = third_channel == 255

    temp_tensor = torch.zeros((512, 512))
    temp_tensor[first_mask & second_mask & third_mask] = 255
    
    return temp_tensor # 512x512


# print(torch.unique(get_binary_change_map(I1)))
# print(get_binary_change_map(I1).size())
# binary_change_map = get_binary_change_map(I1).cpu().numpy().astype(np.uint8)
# image = Image.fromarray(binary_change_map)
# image.save("./wonendie.jpg")


def process_label():
    target_path = "D:/SECOND_train_set/label/"
    origin_path = "D:/SECOND_train_set/label2/"

    file_names = set(os.listdir(origin_path))
    for file_name in tqdm(file_names):
        read_path = origin_path + file_name
        I1 = Image.open(read_path)
        I1 = torch.from_numpy(np.array(I1)).permute(2, 0, 1).float()

        binary_change_map = get_binary_change_map(I1).cpu().numpy().astype(np.uint8)
        counter = np.count_nonzero(binary_change_map == 255)

        image = Image.fromarray(binary_change_map)
        
        # 只保存有 postive_label 的 label
        if counter:
            write_path = target_path + file_name
            image.save(write_path)


def split_train_val_test():
    """
    change shape of SECOND to fixed SECOND -> A, B, label
    bgm: Stranger Thing. OneRepublic/Kygo
    """
    return 
    train_num = 850
    val_num = 283
    test_num = 283
    target_path = "D:/SECOND_train_set/label/"
    file_names = set(os.listdir(target_path))

    train_set = set(random.sample(file_names, train_num))
    file_names = file_names - train_set

    val_set = set(random.sample(file_names, val_num))
    file_names = file_names - val_set

    Second_path = "D:/SECOND_train_set/"
    fixed_Second_path = "D:/SECOND_train_set/Second/"

    folders = ["train", "val", "test"]
    inner_folders = ["A", "B", "label"]

    for folder in folders:
        current_path = Second_path + folder
        for inner_folder in inner_folders:
            _current_path = current_path + '/' + inner_folder
            file_names = set(os.listdir(_current_path))
            for file_name in file_names:
                cur_file = _current_path + '/' + file_name
                another_cur_file = fixed_Second_path + inner_folder + '/' + folder + '_' + file_name
                shutil.copyfile(cur_file, another_cur_file)


def unique_channel_value(I1) -> set:
    """
    对于多分类的 CD Label，查看每个颜色对应的三通道值
    """
    myset = set()

    for i in range(1, 256):
        for j in tqdm(range(1, 256)):
            myset.add(str(I1[0, i, j]) +", "+ str(I1[1, i, j]) + ", " + str(I1[2, i, j]))
            # myset.add(str(I1[i, j]) +", "+ str(I1[i, j]))

    # 0,255,0 亮绿色 树
    # 0,0,255 蓝色 水体
    # 255,255,255 白色 无变化区域
    # 128,0,0 栗色 建筑物
    # 0,128,0 深绿色 Low vegetation
    # 128,128,128 灰色 N.v.g surface
    # 255,0,0 红色 playground

    print(myset)
    return myset


def see_you_feng():
    ori_path = "D:/SECOND_train_set/Second/test/label/"
    source_path = "D:/SECOND_train_set/A/"
    target_path = "D:/SECOND_train_set/Second/test/A/"

    file_names = set(os.listdir(ori_path))

    for file_name in tqdm(file_names):
        shutil.copyfile(source_path + file_name, target_path + file_name)


if __name__ == "__main__":
    # see_you_feng()
    pass