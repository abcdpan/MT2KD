from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

# Z-score标准化
def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        y_s = normalize(y_s)
        y_t = normalize(y_t)

        p_s1 = F.log_softmax(y_s/self.T, dim=1)
        p_t1 = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s1, p_t1, reduction='sum') * (self.T**2) / y_s[0].shape[0]

        return loss


#
# class DistillKL(nn.Module):
#     """Distilling the Knowledge in a Neural Network"""
#     def __init__(self, T):
#         super(DistillKL, self).__init__()
#         self.T = T
#
#     def forward(self, y_s, y_t):
#         y_s = normalize(y_s)
#         y_t = normalize(y_t)
#
#         p_s1 = F.log_softmax(y_s/self.T, dim=1)
#         p_t1 = F.softmax(y_t/self.T, dim=1)
#         loss = F.kl_div(p_s1, p_t1, reduction='sum') * (self.T**2) / y_s[0].shape[0]
#         return loss


