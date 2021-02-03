# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 17:16:57 2019

@author: CS
"""

import os

import torch
from torch import nn
import numpy as np

from tqdm import tqdm
import time

from tifffile import imsave

from skimage.measure import compare_psnr
from cs_util import comput_sigma_from_psnr


def save_checkpoint(opt, model, epoch, model_flag):
    if model_flag == 0:
        dirpath = "VS_Net_model/"
    else:
        dirpath = "GD_Net_model/"

    model_out_path = dirpath + "model_epoch_{}.pkl".format(epoch)
    # check path status
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    # save model
    if (opt.gpu_choose == 1 and (opt.gpu_id == [] or len(opt.gpu_id) == 1)) or (torch.cuda.device_count() <= 1):
        torch.save(model.state_dict(), model_out_path)
    else:
        torch.save(model.module.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def save_img(val_data, epoch, img_class_flag, img_out_dir):
    if not os.path.exists(img_out_dir + "\\"):
        os.makedirs(img_out_dir + "\\")
    if img_class_flag == 0:
        img_class_str = "possion_add_img"
    elif img_class_flag == 1:
        img_class_str = "gassion_add_img"
    elif img_class_flag == 2:
        img_class_str = "noise_val_img"
    elif img_class_flag == 3:
        img_class_str = "clean_val_img"
    elif img_class_flag == 4:
        img_class_str = "VS_gen_img"
    else:
        img_class_str = "GD_gen_img"
    img_out_path = img_out_dir + "\\" + img_class_str + "_epoch_{}.tif".format(epoch)
    val_data = val_data * 255
    imsave(img_out_path, (val_data).astype('uint8'))


# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 00:08:50 2020

@author: CS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 调整学习率
def adjust_learning_rate(optimizer, epoch, opt):
    if not epoch % opt.lr_update_freq and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
    return optimizer

# VST_net 编码器，对泊松噪声进行稳定化
class VS_Net(nn.Module):
    def __init__(self):
        super(VS_Net, self).__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.convSF = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.down = nn.AvgPool2d(2)
        self.conv = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(64,32,3,2,1,1),#batch*32*256*256
            nn.ReLU()
        )
        self.outc = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        conv1 = self.inc(x)
        conv2 = self.convSF(conv1)
        conv3 = self.down(conv2)
        conv4 = self.conv(conv3)
        conv5 = self.up(conv4)
        conv6 = self.convSF(conv5)
        conv7 = self.outc(conv6)
        out = conv7
        return out

# 定义VST_Loss损失
class VS_GD_Loss(nn.Module):
    def __init__(self):
        super(VS_GD_Loss, self).__init__()

    def forward(self, VS_gen_img, GD_gen_img, clean_img, sigma, batchsize):
        diff_img = VS_gen_img - clean_img
        diff_img_min = diff_img.min(1).values.min(1).values.min(1).values
        diff_img_min = diff_img_min.view(batchsize, 1, 1, 1)
        diff_img_min = diff_img_min.expand(batchsize, 1, 512, 512)

        diff_img = diff_img - diff_img_min
        gen_img_everge = diff_img.mean(1).mean(1).mean(1)
        everge_loss = torch.mean(torch.pow((gen_img_everge), 2))

        gen_img_sigma = diff_img.var(2).var(2).view(batchsize)

        sigma_loss = torch.mean(torch.pow((gen_img_sigma - sigma * sigma / 255 / 255), 2))

        contentloss = torch.mean(torch.pow((GD_gen_img - clean_img), 2))

        self.loss = (0.02 * everge_loss + 0.98 * sigma_loss) * 0.4 + contentloss * 0.6
        return self.loss

def train_VS_GD_Net(VS_Net_model, GD_Net_model, opt, trainsize, valsize, train_loader, val_loader,
                           img_out_dir):
    # 设置主GPU  与model = model.cuda(device=opt.gpu_id[0]) # 对于多GPU，可指定GPU作为主GPU一样
    if opt.gpu_id == []:
        GPUstr = "cuda:0"
    else:
        GPUstr = "cuda:" + str(opt.gpu_id[0])

    print("We will run in GPU " + ','.join(str(opt.gpu_id[0])))

    device = torch.device(GPUstr if torch.cuda.is_available() else 'cpu')

    VS_Net_model.to(device)
    GD_Net_model.to(device)

    # GD_Net_model_path = "GD_Net_model/" + "model_epoch_{}.pkl".format(opt.num_epochs - 1)
    GD_Net_model_path = "GD_Net_model/" + "model_epoch_15.pkl"
    GD_Net_model.load_state_dict(torch.load(GD_Net_model_path))

    loss_func = VS_GD_Loss()
    optimizer = torch.optim.Adam(VS_Net_model.parameters(), lr=opt.learning_rate,
                                 weight_decay=opt.lambda_l2)  # Adding L2 regularization

    for param in GD_Net_model.parameters():
        param.requires_grad = False

    # 预定义用来选择最后的权重
    valid_loss_list = []
    best_val_loss = 1000
    best_epoxh = 0
    # 训练模型
    start_time = time.time()
    for epoch in range(opt.num_epochs):
        optimizer = adjust_learning_rate(optimizer, epoch, opt)
        GD_Net_model.train()
        running_loss = 0.0
        print("-" * 10)
        print("Epoch {}/{}".format(epoch, opt.num_epochs))

        for data in tqdm(train_loader):  # tqdm它的作用就是在终端上出现一个进度条
            train_possion_img, train_gassion_img, train_noise_img, train_clean_img, sigma = data

            # 将数据转化为tensor （用DataLoader读入的数据已转化为tensor不需要再转）
            # 将数据放入GPU中 或X_train, y_train = X_train.cuda(device=device_ids[0]),  device_ids[0]可以为device y_train.cuda(device=device_ids[0])
            x_train = train_possion_img.to(device)
            x_train = VS_Net_model(x_train)

            y_train = train_clean_img.to(device)

            # ===================前向传播=====================
            y_pred = GD_Net_model(x_train)
            # 损失
            sigma = sigma.to(device)
            loss = loss_func(x_train, y_pred, y_train, sigma, opt.batch_size)
            # ===================backward====================
            # 梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # ===================更新权重====================
            optimizer.step()
            # ===================打印checkpoint数据========================
            running_loss += loss.data.item()

        GD_Net_model.eval()
        with torch.set_grad_enabled(False):  # save memory during inference
            if epoch % opt.epoch_interval == 0:
                val_loss = 0
                for idxbatch, data in enumerate(val_loader):
                    possion_val, gassion_val, noise_val, clean_val, sigma = data

                    if idxbatch == 1:
                        save_img(possion_val.numpy()[0, 0], epoch, 0, img_out_dir)
                        save_img(gassion_val.numpy()[0, 0], epoch, 1, img_out_dir)
                        save_img(noise_val.numpy()[0, 0], epoch, 2, img_out_dir)
                        save_img(clean_val.numpy()[0, 0], epoch, 3, img_out_dir)

                    x_val = possion_val.to(device)
                    x_val = VS_Net_model(x_val)

                    y_val = clean_val.to(device)

                    y_val_pre = GD_Net_model(x_val)

                    if idxbatch == 1:
                        x_val_pre_save = np.clip(x_val.detach().cpu().numpy()[0, 0], 0, 1).astype(np.float64)
                        save_img(x_val_pre_save, epoch, 4, img_out_dir)
                        y_val_pre_save = np.clip(y_val_pre.detach().cpu().numpy()[0, 0], 0, 1).astype(np.float64)
                        save_img(y_val_pre_save, epoch, 5, img_out_dir)

                    sigma = sigma.to(device)
                    loss = loss_func(x_val, y_val_pre, y_val, sigma, opt.batch_size)
                    val_loss = val_loss + loss

                print("\nEpoch {}/{} | Train Loss is:{:.6f}, Val Loss is:{:.6f}".format(epoch, opt.num_epochs,
                                                                                        running_loss / trainsize,
                                                                                        val_loss / valsize))

                valid_loss_list.append(val_loss / valsize)

        if valid_loss_list[-1] <= best_val_loss:
            best_val_loss = valid_loss_list[-1]
            best_epoxh = epoch

        save_checkpoint(opt, VS_Net_model, epoch, 0)

        elapsed = (time.time() - start_time) / 60
        print(f'Time elapsed: {elapsed:.2f} min')

    print("\nIn epoch {}, we get the minist Val loss is:{:.6f}".format(best_epoxh, best_val_loss))

    elapsed = (time.time() - start_time) / 60
    print(f'Total Training Time: {elapsed:.2f} min')

    return VS_Net_model, valid_loss_list
