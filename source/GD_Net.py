# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 00:08:50 2020

@author: CS
"""

import os

import numpy as np

from tqdm import tqdm
import time

from tifffile import imsave

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    else:
        img_class_str = "Net_gen_img"
    img_out_path = img_out_dir + "\\" + img_class_str + "_epoch_{}.tif".format(epoch)
    val_data = val_data * 255
    imsave(img_out_path, (val_data).astype('uint8'))

# 调整学习率
def adjust_learning_rate(optimizer, epoch, opt):
    if not epoch % opt.lr_update_freq and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5
    return optimizer

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  #nn.AdaptiveAvgPool2d(output_size)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# 去噪网络
class GD_Net(nn.Module):
    def __init__(self):
        super(GD_Net, self).__init__()
        
        self.inc = nn.Sequential(
            single_conv(1, 64),
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            CALayer(256, 16),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.outc = outconv(64, 1)

    def forward(self, x):
        inx = self.inc(x)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        
        return out

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2))  # 对x1进行填充，使其大小与x2一样

        x = x2 + x1
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

## denoise_loss
class GD_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, out_image, gt_image):
        loss = torch.mean(torch.pow((out_image - gt_image), 2))
        return loss

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

def train_GD_Net(model, opt, trainsize, valsize, train_loader, val_loader,
                           img_out_dir):
    # 设置主GPU  与model = model.cuda(device=opt.gpu_id[0]) # 对于多GPU，可指定GPU作为主GPU一样
    if opt.gpu_id == []:
        GPUstr = "cuda:0"
    else:
        GPUstr = "cuda:" + str(opt.gpu_id[0])

    print("We will run in GPU " + ','.join(str(opt.gpu_id[0])))

    device = torch.device(GPUstr if torch.cuda.is_available() else 'cpu')

    model.to(device)

    loss_func = GD_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate,
                                 weight_decay=opt.lambda_l2)  # Adding L2 regularization

    # 预定义用来选择最后的权重
    valid_loss_list = []
    best_val_loss = 1000
    best_epoxh = 0
    # 训练模型
    start_time = time.time()
    for epoch in range(opt.num_epochs):
        running_loss = 0.0
        optimizer = adjust_learning_rate(optimizer, epoch, opt)
        model.train()
        print("-" * 10)
        print("Epoch {}/{}".format(epoch, opt.num_epochs))

        for data in tqdm(train_loader):  # tqdm它的作用就是在终端上出现一个进度条
            train_possion_img, train_gassion_img, train_noise_img, train_clean_img, sigma = data

            # 将数据转化为tensor （用DataLoader读入的数据已转化为tensor不需要再转）
            # 将数据放入GPU中 或X_train, y_train = X_train.cuda(device=device_ids[0]),  device_ids[0]可以为device y_train.cuda(device=device_ids[0])
            x_train = train_gassion_img.to(device)
            y_train = train_clean_img.to(device)

            # ===================前向传播=====================
            y_pred = model(x_train)
            # 损失
            loss = loss_func(y_pred, y_train)
            # ===================backward====================
            # 梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # ===================更新权重====================
            optimizer.step()
            # ===================打印checkpoint数据========================
            running_loss += float(loss.data.item())

        model.eval()
        with torch.set_grad_enabled(False):  # save memory during inference
            if epoch % opt.epoch_interval == 0:
                val_loss = 0
                for idxbatch, data in enumerate(val_loader):
                    possion_val, gassion_val, noise_val, clean_val, sigma_val = data

                    if idxbatch == 1:
                        save_img(gassion_val.numpy()[0, 0], epoch, 1, img_out_dir)
                        save_img(clean_val.numpy()[0, 0], epoch, 3, img_out_dir)

                    x_val = gassion_val.to(device)
                    y_val = clean_val.to(device)
                    y_val_pre = model(x_val)

                    if idxbatch == 1:
                        y_val_pre_save = np.clip(y_val_pre.detach().cpu().numpy()[0, 0], 0, 1).astype(np.float64)
                        save_img(y_val_pre_save, epoch, 4, img_out_dir)

                    loss = loss_func(y_val_pre, y_val)
                    val_loss = val_loss + loss

                print("\nEpoch {}/{} | Train Loss is:{:.6f}, Val Loss is:{:.6f}".format(epoch, opt.num_epochs,
                                                                                        running_loss / trainsize,
                                                                                        val_loss / valsize))

                valid_loss_list.append(val_loss / valsize)

        if valid_loss_list[-1] <= best_val_loss:
            best_val_loss = valid_loss_list[-1]
            best_epoxh = epoch

        save_checkpoint(opt, model, epoch, 1)

        elapsed = (time.time() - start_time) / 60
        print(f'Time elapsed: {elapsed:.2f} min')

    print("\nIn epoch {}, we get the minist Val loss is:{:.6f}".format(best_epoxh, best_val_loss))

    elapsed = (time.time() - start_time) / 60
    print(f'Total Training Time: {elapsed:.2f} min')

    return model, valid_loss_list


def train_VISD_Net(VS_Net_model, GD_Net_model, opt, trainsize, valsize, train_loader, val_loader,
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

    VS_Net_model_path = "VS_Net_model/" + "model_epoch_49.pkl"
    VS_Net_model.load_state_dict(torch.load(VS_Net_model_path))

    loss_func = GD_Loss()
    optimizer = torch.optim.Adam(GD_Net_model.parameters(), lr=opt.learning_rate,
                                 weight_decay=opt.lambda_l2)  # Adding L2 regularization

    for param in VS_Net_model.parameters():
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
            loss = loss_func(y_pred, y_train)
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
                        y_val_pre_save = np.clip(y_val_pre.detach().cpu().numpy()[0, 0], 0, 1).astype(np.float64)
                        save_img(y_val_pre_save, epoch, 4, img_out_dir)

                    loss = loss_func(y_val_pre, y_val)
                    val_loss = val_loss + loss

                print("\nEpoch {}/{} | Train Loss is:{:.6f}, Val Loss is:{:.6f}".format(epoch, opt.num_epochs,
                                                                                        running_loss / trainsize,
                                                                                        val_loss / valsize))

                valid_loss_list.append(val_loss / valsize)

        if valid_loss_list[-1] <= best_val_loss:
            best_val_loss = valid_loss_list[-1]
            best_epoxh = epoch

        save_checkpoint(opt, GD_Net_model, epoch, 1)

        elapsed = (time.time() - start_time) / 60
        print(f'Time elapsed: {elapsed:.2f} min')

    print("\nIn epoch {}, we get the minist Val loss is:{:.6f}".format(best_epoxh, best_val_loss))

    elapsed = (time.time() - start_time) / 60
    print(f'Total Training Time: {elapsed:.2f} min')

    return GD_Net_model, valid_loss_list


