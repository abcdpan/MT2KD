from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VIDLoss(nn.Module):
    def __init__(self,num_input_channels,num_mid_channel,num_target_channels,init_pred_var=5.0,eps=1e-5):
        super(VIDLoss, self).__init__()
        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels,kernel_size=1, padding=0,bias=False, stride=stride)
        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var-eps)-1.0) * torch.ones(num_target_channels)
            )
        self.eps = eps
    def forward(self, input, target):
        s_H, t_H = input.shape[2], target.shape[2]  # 为了维度匹配进行池化
        if s_H > t_H:
            input = F.adaptive_avg_pool2d(input, (t_H, t_H))
        elif s_H < t_H:
            target = F.adaptive_avg_pool2d(target, (s_H, s_H))
        else:            pass
        pred_mean = self.regressor(input)  # 输入经过3个1x1回归器。
        pred_var = torch.log(1.0+torch.exp(self.log_scale))+self.eps # 信息熵公式
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5*(
            (pred_mean-target)**2/pred_var+torch.log(pred_var)
            )
        loss = torch.mean(neg_log_prob)
        return loss
