# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:38:05 2018

@author: Administer
"""

import os
import numpy as np
from tifffile import imsave, imread
import random
import shutil

from skimage.measure import compare_psnr


def rename(img_dir):
    filelist = os.listdir(img_dir)  # 获取文件路径
    i = 1  # 表示文件的命名是从1开始的
    for item in filelist:
        if item.endswith('.tif'):  # 初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
            newname = item.split('_')
            newname = newname[-1]
            src = os.path.join(os.path.abspath(img_dir), item)
            dst = os.path.join(os.path.abspath(img_dir), newname)  # 处理后的格式也为jpg格式的，当然这里可以改成png格式
            os.rename(src, dst)
            i = i + 1


def read_cut_smal_img(img_source_dir, endword, img_target_dir):
    for filename in os.listdir(img_source_dir):
        if filename.endswith(endword):
            img = imread(img_source_dir + "\\" + filename)
            # 反色
            img = 255 - img

            img_source_num = filename.split('_')
            img_source_num = img_source_num[-1]
            img_source_num = img_source_num.split('.')
            img_source_num = img_source_num[0]

            patchnum = 0
            for y in range(0, img.shape[0] - 512 + 1, 512):  # 跳的是shape大小的，patch之间是没有重叠的
                for x in range(0, img.shape[1] - 512 + 1, 512):
                    smallpatch = img[y:y + 512, x:x + 512]
                    patchnumstr = "%03d" % patchnum
                    img_path = img_target_dir + "\\" + img_source_num + "_" + patchnumstr + ".tif"
                    imsave(img_path, (smallpatch).astype('uint8'))
                    patchnum = patchnum + 1


def random_select_img(img_source_dir, img_target_dir, imgnum):
    pathDir = os.listdir(img_source_dir)
    sample = random.sample(pathDir, imgnum)
    # print(sample)
    for name in sample:
        shutil.move(img_source_dir + name, img_target_dir + name)

def random_select_clean_noise(noiseimg_source_dir, noiseimg_target_dir, cleanimg_source_dir, cleanimg_target_dir,
                              imgnum):
    pathDir = os.listdir(noiseimg_source_dir)
    sample = random.sample(pathDir, imgnum)
    # print(sample)
    for name in sample:
        shutil.move(noiseimg_source_dir + "\\" + name, noiseimg_target_dir + "\\" + name)
        shutil.move(cleanimg_source_dir + "\\" + name, cleanimg_target_dir + "\\" + name)


def move_img(img_source_dir, img_target_dir):
    pathDir = os.listdir(img_source_dir)
    for name in pathDir:
        shutil.move(img_source_dir + "\\" + name, img_target_dir + "\\" + name)


def move_clean_noise(noiseimg_source_dir, noiseimg_target_dir, cleanimg_source_dir, cleanimg_target_dir):
    pathDir = os.listdir(noiseimg_source_dir)
    for name in pathDir:
        shutil.move(noiseimg_source_dir + "\\" + name, noiseimg_target_dir + "\\" + name)
        shutil.move(cleanimg_source_dir + "\\" + name, cleanimg_target_dir + "\\" + name)


def rotate_dataset(img_source_dir, endword, img_target_dir):
    for filename in os.listdir(img_source_dir):
        if filename.endswith(endword):
            imgname = filename.split('.')
            imgname = imgname[0]
            img = imread(img_source_dir + "\\" + filename)
            out = np.rot90(img, 1)
            img_path = img_target_dir + "\\" + imgname + "_1" + ".tif"
            imsave(img_path, (out).astype('uint8'))
            out = np.rot90(img, 2)
            img_path = img_target_dir + "\\" + imgname + "_2" + ".tif"
            imsave(img_path, (out).astype('uint8'))
            out = np.rot90(img, 3)
            img_path = img_target_dir + "\\" + imgname + "_3" + ".tif"
            imsave(img_path, (out).astype('uint8'))

            img = np.flipud(img)
            img_path = img_target_dir + "\\" + imgname + "_4" + ".tif"
            imsave(img_path, (img).astype('uint8'))

            out = np.rot90(img, 1)
            img_path = img_target_dir + "\\" + imgname + "_5" + ".tif"
            imsave(img_path, (out).astype('uint8'))
            out = np.rot90(img, 2)
            img_path = img_target_dir + "\\" + imgname + "_6" + ".tif"
            imsave(img_path, (out).astype('uint8'))
            out = np.rot90(img, 3)
            img_path = img_target_dir + "\\" + imgname + "_7" + ".tif"
            imsave(img_path, (out).astype('uint8'))


def possion_noise_factor(img, possion_factor=255):
    img_possion_noise = np.clip(np.random.poisson(img), 0, possion_factor)
    return img_possion_noise

def comput_sigma_from_psnr(imgpsnr):
    gassion_sigama = 255 / (np.power(10, imgpsnr / 20))
    return gassion_sigama


def noise_dataset(img_source_dir, endword, img_SEMnoise_dir, img_gassion_dir):
    for filename in os.listdir(img_source_dir):
        if filename.endswith(endword):
            imgname = filename.split('.')
            imgname = imgname[0]
            img = imread(img_source_dir + "\\" + filename)

            ##
            atemp = 1 / 0.1158
            ##
            clean_img_temp = img * atemp
            img_SEMnoise = np.random.poisson(clean_img_temp)
            img_SEMnoise = img_SEMnoise / atemp
            img_SEMnoise = np.clip(img_SEMnoise, 0, 255)

            possion_factor = np.random.randint(low=20, high=100)
            img_SEMnoise = img_SEMnoise / 255 * possion_factor
            img_SEMnoise = possion_noise_factor(img_SEMnoise, possion_factor)
            img_SEMnoise = img_SEMnoise / possion_factor * 255
            ##
            img_SEMnoise = img_SEMnoise - np.random.normal(0, 0.7298, img.shape)
            img_SEMnoise = np.clip(img_SEMnoise, 0, 255)
            ##
            SEMnoise_img_path = img_SEMnoise_dir + "\\" + imgname + ".tif"
            imsave(SEMnoise_img_path, (img_SEMnoise).astype('uint8'))

            img_psnr = compare_psnr(img_SEMnoise, img, 255)
            gassion_sigma = comput_sigma_from_psnr(img_psnr)
            img_gassion = np.clip(img + np.random.normal(0, gassion_sigma, img.shape), 0, 255)

            gassion_img_path = img_gassion_dir + "\\" + imgname + ".tif"
            imsave(gassion_img_path, (img_gassion).astype('uint8'))


def text_save(filename, data):
    file = open(filename, 'w')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("\nFile saved successfully!")

