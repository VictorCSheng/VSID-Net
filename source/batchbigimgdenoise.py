# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 15:31:03 2020

@author: CS
"""

import numpy as np

from tifffile import imsave,imread
import os 

initial_directory = 'H:\\data\\copy'  #图像命名方式01，02
inter_directory="H:\\data\\VSIDdenoise\\smallimg"
final_directory="H:\\data\\VSIDdenoise\\denoisesmallimg"
finalbig_directory = 'H:\\data\\VSIDdenoise\\final'

sizelist = []

for filename in os.listdir(initial_directory):
    filestr = filename.split('.')
    filenamepre = filestr[0]
    print(filenamepre)  # 仅仅是为了测试
    image = imread(initial_directory + "//" + filename)

    sizelist.append([np.size(image, 0), np.size(image, 1)])

    constsize = 512
    offsize = 10
    imagey = 0
    patchnum = 0
    while (imagey + constsize <= image.shape[0]):
        imagex = 0
        while (imagex + constsize <= image.shape[1]):
            smallpatch = image[imagey:imagey + constsize, imagex:imagex + constsize]
            patchnumstr = "%03d" % patchnum
            img_path = inter_directory + "\\" + filenamepre + "." + patchnumstr + ".tif"
            imsave(img_path, (smallpatch).astype('uint8'))
            patchnum = patchnum + 1

            imagex = imagex + constsize - offsize

        if (imagex < image.shape[1]):
            smallpatch = image[imagey:imagey + constsize, image.shape[1] - constsize:image.shape[1]]
            patchnumstr = "%03d" % patchnum
            img_path = inter_directory + "\\" + filenamepre + "." + patchnumstr + ".tif"
            imsave(img_path, (smallpatch).astype('uint8'))
            patchnum = patchnum + 1

        imagey = imagey + constsize - offsize

    if (imagey < image.shape[0]):
        imagex = 0
        while (imagex + constsize <= image.shape[1]):
            smallpatch = image[image.shape[0] - constsize:image.shape[0], imagex:imagex + constsize]
            patchnumstr = "%03d" % patchnum
            img_path = inter_directory + "\\" + filenamepre + "." + patchnumstr + ".tif"
            imsave(img_path, (smallpatch).astype('uint8'))
            patchnum = patchnum + 1

            imagex = imagex + constsize - offsize

        if (imagex < image.shape[1]):
            smallpatch = image[image.shape[0] - constsize:image.shape[0], image.shape[1] - constsize:image.shape[1]]
            patchnumstr = "%03d" % patchnum
            img_path = inter_directory + "\\" + filenamepre + "." + patchnumstr + ".tif"
            imsave(img_path, (smallpatch).astype('uint8'))
            patchnum = patchnum + 1


from VS_Net import VS_Net
from GD_Net import GD_Net
import torch

from skimage import img_as_float

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

VS_Net_model = VS_Net()
GD_Net_model = GD_Net()
VS_Net_model = VS_Net_model.to(device)
GD_Net_model = GD_Net_model.to(device)

VS_Net_path = 'H:\\vstdenoise_result\\VSID_net\\VS_Net_model\\model_epoch_49.pkl'
GD_Net_path = 'H:\\vstdenoise_result\\VSID_net\\GD_Net_model\\model_epoch_68.pkl'   # 0

VS_Net_model.load_state_dict(torch.load(VS_Net_path))
GD_Net_model.load_state_dict(torch.load(GD_Net_path))

endword = 'tif'

testpoissonimg = inter_directory
poisson_Clean_img = final_directory

for filename in os.listdir(testpoissonimg):
    if filename.endswith(endword):
        poissonimg = imread(testpoissonimg + "\\" + filename)

        poissonimg = img_as_float(poissonimg)
        poissonimg = torch.Tensor(poissonimg[np.newaxis, np.newaxis])
        poissonimg = poissonimg.to(device)

        Gen_Gaussion_img = VS_Net_model(poissonimg)

        Gen_Clean_img_save = GD_Net_model(Gen_Gaussion_img).detach().cpu()

        Gen_Clean_img_save = np.clip(np.squeeze(Gen_Clean_img_save.numpy()[0]), 0, 1) * 255

        imsave(poisson_Clean_img + '\\' + filename, (Gen_Clean_img_save).astype('uint8'))


from math import floor, ceil
filenum=1
patchnum=0
# sizelist=[[12287,12288],[12288,12288],[12287,12288],[12288,12288],[12288,12288],[12287,12288],[12288,12288],[12288,12288],[12285,12288],[12269,12288]]
imgfinl=np.zeros((sizelist[filenum-1][0],sizelist[filenum-1][1]))
constsize = 512
offsize = 10
# counstvalrow = ceil(sizelist[filenum-1][0]/(constsize - offsize))
# counstvalcol = ceil(sizelist[filenum-1][1]/(constsize - offsize))
for filename in os.listdir(final_directory):
    print(filename)  # 仅仅是为了测试
    counstvalrow = ceil(sizelist[filenum - 1][0] / (constsize - offsize))
    counstvalcol = ceil(sizelist[filenum - 1][1] / (constsize - offsize))
    filestr = filename.split('.')
    if filenum == int(filestr[0]):
        row = floor(patchnum / counstvalrow)
        col = patchnum % counstvalrow
        image = imread(final_directory + "//" + filename)
        if row == 0:
            if col == 0:
                imgfinl[0:507, 0:507] = image[0:507, 0:507]
                patchnum = patchnum + 1
            elif col == (counstvalcol - 1):
                imgfinl[0:507, sizelist[filenum-1][1] - 502:sizelist[filenum-1][1]] = image[0:507, 10:512]
                patchnum = patchnum + 1
            else:
                imgfinl[0:507, (col - 1) * 502 + 507:col * 502 + 507] = image[0:507, 5:507]
                patchnum = patchnum + 1

        elif row == (counstvalrow - 1):
            if col == 0:
                imgfinl[sizelist[filenum-1][0] - 502:sizelist[filenum-1][0], 0:507] = image[10:512, 0:507]
                patchnum = patchnum + 1
            elif col == (counstvalcol - 1):
                imgfinl[sizelist[filenum-1][0] - 502:sizelist[filenum-1][0], sizelist[filenum-1][1] - 502:sizelist[filenum-1][1]] = image[10:512, 10:512]
                patchnum = patchnum + 1
            else:
                imgfinl[sizelist[filenum-1][0] - 502:sizelist[filenum-1][0], (col - 1) * 502 + 507:col * 502 + 507] = image[10:512, 5:507]
                patchnum = patchnum + 1

        else:
            if col == 0:
                imgfinl[(row - 1) * 502 + 507:row * 502 + 507, 0:507] = image[5:507, 0:507]
                patchnum = patchnum + 1
            elif col == (counstvalcol - 1):
                imgfinl[(row - 1) * 502 + 507:row * 502 + 507, sizelist[filenum-1][1] - 502:sizelist[filenum-1][1]] = image[5:507, 10:512]
                patchnum = patchnum + 1
            else:
                imgfinl[(row - 1) * 502 + 507:row * 502 + 507, (col - 1) * 502 + 507:col * 502 + 507] = image[5:507,5:507]
                patchnum = patchnum + 1
    else:
        imsave(finalbig_directory + "\\0" + str(filenum) + ".tif", (imgfinl).astype('uint8'))
        patchnum = 0
        filenum = filenum + 1
        imgfinl = np.zeros((sizelist[filenum-1][0], sizelist[filenum-1][1]))

        image = imread(final_directory + "//" + filename)
        imgfinl[0:507, 0:507] = image[0:507, 0:507]
        patchnum = patchnum + 1

imsave(finalbig_directory + "\\0" + str(filenum) + ".tif", (imgfinl).astype('uint8'))
