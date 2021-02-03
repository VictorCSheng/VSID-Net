# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 18:49:28 2019

@author: CS
"""

from torch.utils.data import Dataset
from os.path import join
from glob import glob
from tifffile import imread, imsave

import numpy as np

from skimage.measure import compare_psnr

from cs_util import comput_sigma_from_psnr


class Dataset_net(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, noise_dir, clean_dir, SEMnoise_dir, gassion_dir, filter='*.tif', transform=None):  # 初始化一些需要传入的参数
        super(Dataset_net, self).__init__()

        noise_imgs = glob(join(noise_dir, filter))
        noise_imgs.sort()
        self.noise_imgs = noise_imgs

        clean_imgs = glob(join(clean_dir, filter))
        clean_imgs.sort()
        self.clean_imgs = clean_imgs

        SEMnoise_imgs = glob(join(SEMnoise_dir, filter))
        SEMnoise_imgs.sort()
        self.SEMnoise_imgs = SEMnoise_imgs

        gassion_imgs = glob(join(gassion_dir, filter))
        gassion_imgs.sort()
        self.gassion_imgs = gassion_imgs

        self.transform = transform

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.clean_imgs)

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        noise_img_path = self.noise_imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        noise_img = imread(noise_img_path)  # 按照path读入图片from PIL import Image # 按照路径读取图片

        clean_img_path = self.clean_imgs[index]
        clean_img = imread(clean_img_path)

        SEMnoise_img_path = self.SEMnoise_imgs[index]
        SEMnoise_img = imread(SEMnoise_img_path)

        gassion_img_path = self.gassion_imgs[index]
        gassion_img = imread(gassion_img_path)

        img_psnr = compare_psnr(SEMnoise_img, clean_img, 255)
        gassion_sigma = comput_sigma_from_psnr(img_psnr)

        if self.transform is not None:
            SEMnoise_img = self.transform(SEMnoise_img)
            gassion_img = self.transform(gassion_img)

            noise_img = self.transform(noise_img)
            clean_img = self.transform(clean_img)

        return SEMnoise_img, gassion_img, noise_img, clean_img, gassion_sigma  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

