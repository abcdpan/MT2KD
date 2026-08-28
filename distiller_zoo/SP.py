from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class Similarity(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, g_s, g_t):
        return [self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1) #[64,256*8*8]
        f_t = f_t.view(bsz, -1) #[64,256*8*8]

        G_s = torch.mm(f_s, torch.t(f_s))        # [64,256*8*8] 乘 [256*8*8,64] = [64,64]
        G_s = torch.nn.functional.normalize(G_s) # 归一化
        G_t = torch.mm(f_t, torch.t(f_t))        # [64,256*8*8] 乘 [256*8*8,64] = [64,64]
        G_t = torch.nn.functional.normalize(G_t) # 归一化

        G_diff = G_t - G_s #[64,64]
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz) #[64,64]*[64,64] -> [1] / (64*64)
        return loss
