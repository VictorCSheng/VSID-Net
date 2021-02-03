# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 08:47:14 2019

@author: CS
"""

import os

from cs_util import read_cut_smal_img, random_select_clean_noise, move_clean_noise, rotate_dataset, noise_dataset, rename

import random
random.seed(0)

ini_noise_source_dir = "../oriimg/noise"
ini_clean_source_dir = "../oriimg/clean"

ini_noise_target_dir = "../ini/noise"
ini_clean_target_dir = "../ini/clean"

train_noise_target_dir = "../train/noise"
train_clean_target_dir = "../train/clean"
train_img_SEMnoise_dir = "../train/SEMnoise"
train_img_gassion_dir = "../train/gassion"

val_noise_target_dir = "../val/noise"
val_clean_target_dir = "../val/clean"
val_img_SEMnoise_dir = "../val/SEMnoise"
val_img_gassion_dir = "../val/gassion"

test_noise_target_dir = "../test/noise"
test_clean_target_dir = "../test/clean"
test_img_SEMnoise_dir = "../test/SEMnoise"
test_img_gassion_dir = "../test/gassion"

endword = 'tif'

## 对图片重命名，方便后面的切割图像对应
rename(ini_noise_source_dir)
rename(ini_clean_source_dir)

# 切割图片
read_cut_smal_img(ini_noise_source_dir, endword, ini_noise_target_dir)
read_cut_smal_img(ini_clean_source_dir, endword, ini_clean_target_dir)

# 确定小图的数量
pathDir = os.listdir(ini_noise_target_dir)
filenumber = len(pathDir)

## 确定用来训练，校验，测试的图片数量
# 训练集
trainrate = 0.7  # 自定义抽取图片的比例
picknumber = int(filenumber * trainrate)  # 按照rate比例从文件夹中取一定数量图片
random_select_clean_noise(ini_noise_target_dir, train_noise_target_dir, ini_clean_target_dir, train_clean_target_dir, picknumber)

# 校验集
valrate = 0.2
picknumber = int(filenumber * valrate)  # 按照rate比例从文件夹中取一定数量图片
random_select_clean_noise(ini_noise_target_dir, val_noise_target_dir, ini_clean_target_dir, val_clean_target_dir, picknumber)

# 测试集
move_clean_noise(ini_noise_target_dir, test_noise_target_dir, ini_clean_target_dir, test_clean_target_dir)

# 对图像进行旋转和翻转，扩充数据集
# 训练集
rotate_dataset(train_noise_target_dir, endword, train_noise_target_dir)
rotate_dataset(train_clean_target_dir, endword, train_clean_target_dir)
noise_dataset(train_clean_target_dir, endword, train_img_SEMnoise_dir, train_img_gassion_dir)
# 校验集
rotate_dataset(val_noise_target_dir, endword, val_noise_target_dir)
rotate_dataset(val_clean_target_dir, endword, val_clean_target_dir)
noise_dataset(val_clean_target_dir, endword, val_img_SEMnoise_dir, val_img_gassion_dir)
# 测试集
noise_dataset(test_clean_target_dir, endword, test_img_SEMnoise_dir, test_img_gassion_dir)
