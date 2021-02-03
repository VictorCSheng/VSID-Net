# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:20:13 2019

@author: CS
"""

###########################训练模型
# 模型参数设置
import argparse

import torch
import numpy as np
# 参数传入并赋值
# gpu_choose = 1 并且gpu_id = [] 为单GPU运行
parser = argparse.ArgumentParser(description='Train CNN Models')
parser.add_argument('--gpu_choose', default=1, type=int, help='If you choose a specific GPU, set this number to 1 and specify the gpu_id below, otherwise all available GPUs will be used')
parser.add_argument('--gpu_id', default=[0], type=list, help='specified gpu ids you want to use')
parser.add_argument('--num_epochs', default=2, type=int, help='train epoch number') # 20
parser.add_argument('--epoch_interval', default=1, type=int, help='the number of several epochs show a loss')
parser.add_argument('--batch_size', default=5, type=int, help='batch_size')   #100 25
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--learning_rate', default=1e-3, type=int, help='learning_rate')
parser.add_argument('--lr_update_freq', default=20, type=int, help='learning_rate_update_frequency')
parser.add_argument('--dropout_p', default=0.4, type=int, help='the ratio of dropout')
parser.add_argument('--lambda_l2', default=1e-4, type=int, help='L2 regularization of weights')
parser.add_argument('--weights_path', default='params.pkl', type=str, help='weight storage path')

opt = parser.parse_args()
# 设置随机种子来保证结果可复现
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

#########################放入模型的数据集生成
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from Dataset_prepare import Dataset_net

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

data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

train_img = Dataset_net(train_noise_target_dir, train_clean_target_dir, train_img_SEMnoise_dir, train_img_gassion_dir, transform=data_transforms)
val_img = Dataset_net(val_noise_target_dir, val_clean_target_dir, val_img_SEMnoise_dir, val_img_gassion_dir, transform=data_transforms)
print("We have %i train samples and %i val samples." % (len(train_img), len(val_img)))

train_loader = DataLoader(train_img, batch_size=opt.batch_size,shuffle=True,pin_memory=True,drop_last=True)
val_loader = DataLoader(val_img, batch_size=opt.batch_size,shuffle=False,pin_memory=True,drop_last=True)
print("The batch size is %i, the train_loader size is %i and the val_loader size is %i." % (opt.batch_size, len(train_loader), len(val_loader)))

from cs_util import text_save

#### 三阶段训练网络
## 第一阶段：粗训练GD_Net
from GD_Net import GD_Net, train_GD_Net
# GD_Net0_model = GD_Net()
# GD_Net0_img_out_dir = "../train_temp_img/GD_Net0"
# GD_Net0_historymodel, GD_Net0_valid_loss_list = train_GD_Net(GD_Net0_model, opt, len(train_img), len(val_img), train_loader, val_loader, GD_Net0_img_out_dir)
# text_save('./losstxt/GD_Net0_valid_loss_list.txt', GD_Net0_valid_loss_list)

## 第二阶段：在粗训练的GD_Net的基础上训练VS_Net
from VS_Net import VS_Net, train_VS_GD_Net
# GD_Net0_model = GD_Net()
# VS_Net_model = VS_Net()
# VS_Net_img_out_dir = "../train_temp_img/VS_Net"
# VS_Net_historymodel, VS_Net_valid_loss_list = train_VS_GD_Net(VS_Net_model, GD_Net0_model, opt, len(train_img), len(val_img), train_loader, val_loader, VS_Net_img_out_dir)
# text_save('./losstxt/VS_Net_valid_loss_list.txt', VS_Net_valid_loss_list)

## 第三阶段：在VS_Net的基础上重新训练GD_Net
from GD_Net import GD_Net, train_VISD_Net
VS_Net_model = VS_Net()
GD_Net_model = GD_Net()
GD_Net_img_out_dir = "../train_temp_img/GD_Net"
GD_Net_historymodel, GD_Net_valid_loss_list = train_VISD_Net(VS_Net_model, GD_Net_model, opt, len(train_img), len(val_img), train_loader, val_loader, GD_Net_img_out_dir)
text_save('GD_Net_valid_loss_list.txt', GD_Net_valid_loss_list)







