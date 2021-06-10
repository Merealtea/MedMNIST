# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 10:16:35 2021

@author: Yuhao Wang Xingyuan Chen
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

class res_unit50(nn.Module):
    def __init__(self,in_channel,out_channel,stride):
        super(res_unit50,self).__init__()
        self.Sequential = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size = 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel, kernel_size = 3,padding = 1,stride = stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel,4* out_channel, kernel_size=1),
            nn.BatchNorm2d(4*out_channel)
            )
        self.shortcut = nn.Sequential()
        if(stride != 1 or in_channel != 4*out_channel):
            self.shortcut.add_module('conv',module = nn.Conv2d(in_channel, 4 * out_channel,
                                          kernel_size = 1,stride = stride))
            self.shortcut.add_module('bn',module=nn.BatchNorm2d(4*out_channel))
            
    def forward(self,data):
        out = self.Sequential(data)
        out = out + self.shortcut(data)
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.shortcut = nn.Sequential()
        if stride != 1: 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Resnet(nn.Module):
    def __init__(self,config):
        super(Resnet,self).__init__()
        
        """
        Parameters
        ----------
        config : list
            第一个数表示图片的channel数
            第二个数表示是二分类还是多分类还是回归/是哪一种器官
            
            第三个数表示是否为multilabel-binaryclass
            第四个数表示用res-18 还是 res-50
        """
        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)
        self.in_channel = config[0]
        self.task = config[1]
        self.conv = nn.Conv2d(self.in_channel, 64, 3,padding = 1)
        self.bn = nn.BatchNorm2d(64)
        self.activation = nn.ReLU()
        if config[2]:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        if config[3] == 18:
            self.fc_channel = 512
            self.block1 = self.res_unit18(2,[1,1],64,64,3)
            self.block2 = self.res_unit18(2,[2,1],64,128,3)
            self.block3 = self.res_unit18(2,[2,1],128,256,3)
            self.block4 = self.res_unit18(2,[2,1],256,512,3)
            # 你的res18，记得在里面加上激活函数
        else:
            self.fc_channel = 2048
            self.block1 = nn.Sequential(
                res_unit50(64,64,1),
                res_unit50(256,64,1),
                res_unit50(256,64,1)
                )
            self.block2 = nn.Sequential(
                res_unit50(256,128,2),
                res_unit50(512,128,1),
                res_unit50(512,128,1),
                res_unit50(512,128,1)
                )
            self.block3 = nn.Sequential(
                res_unit50(512,256,2),
                res_unit50(1024,256,1),
                res_unit50(1024,256,1),
                res_unit50(1024,256,1),
                res_unit50(1024,256,1),
                res_unit50(1024,256,1)
                )
            self.block4 = nn.Sequential(
                res_unit50(1024,512,2),
                res_unit50(2048,512,1),
                res_unit50(2048,512,1)
                )
        self.avgpool = nn.AvgPool2d(4)
        
        self.fc = nn.Linear(self.fc_channel, self.task) 

    
    def forward(self,data):
        
        out = self.conv(data)
        out = self.bn(out)
        out = self.activation(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.avgpool(out)
        out = out.view(out.shape[0],-1)
        out = self.fc(out)
        return out
    
    def res_unit18(self,num_block,stride,in_channel,out_channel,kernel_size):
        
        layer = []
        for i in stride:
            layer.append(BasicBlock(in_channel, out_channel,i))
            in_channel = in_channel * i
        return nn.Sequential(*layer)
    