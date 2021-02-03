# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:20:13 2019

@author: CS
"""

from VS_Net import VS_Net
from GD_Net import GD_Net
import torch
import numpy as np
import os

from tifffile import imsave,imread
from skimage import img_as_float

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

VS_Net_model = VS_Net()
GD_Net_model = GD_Net()
VS_Net_model = VS_Net_model.to(device)
GD_Net_model = GD_Net_model.to(device)

VS_Net_path = './VS_Net_model/model_epoch_49.pkl'
GD_Net_path = './GD_Net_model/model_epoch_68.pkl'

VS_Net_model.load_state_dict(torch.load(VS_Net_path))
GD_Net_model.load_state_dict(torch.load(GD_Net_path))

endword = 'tif'

testSEMnoiseimg = "../test/SEMnoise/2_023.tif"
SEMnoise_Gaussion_img = "../test/result/gassion/2_023.tif"
SEMnoise_Clean_img = "../test/result/clean/2_023.tif"

SEMnoiseimg = imread(testSEMnoiseimg)

SEMnoiseimg = img_as_float(SEMnoiseimg)
SEMnoiseimg = torch.Tensor(SEMnoiseimg[np.newaxis, np.newaxis])
SEMnoiseimg = SEMnoiseimg.to(device)

Gen_Gaussion_img = VS_Net_model(SEMnoiseimg)

Gen_Gaussion_img_save = Gen_Gaussion_img.detach().cpu()
Gen_Clean_img_save = GD_Net_model(Gen_Gaussion_img).detach().cpu()

Gen_Gaussion_img_save = np.clip(np.squeeze(Gen_Gaussion_img_save.numpy()[0]), 0, 1) * 255
Gen_Clean_img_save = np.clip(np.squeeze(Gen_Clean_img_save.numpy()[0]), 0, 1) * 255

imsave(SEMnoise_Gaussion_img, (Gen_Gaussion_img_save).astype('uint8'))

imsave(SEMnoise_Clean_img, (Gen_Clean_img_save).astype('uint8'))