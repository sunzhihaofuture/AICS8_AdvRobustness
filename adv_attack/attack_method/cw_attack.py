import torch, os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

def cw_l2_attack(model,
                 images,
                 labels,
                 targeted=False,
                 c=0.1,
                 kappa=0,
                 max_iter=1000,
                 learning_rate=0.01,
                 device=None):

    # Define f-function
    def f(x):
        # 论文中的 Z(X) 输出 batchsize, num_classes
        outputs = model(x)
        # batchszie，根据labels 的取值，确定每一行哪一个为 1
        # >>> a = torch.eye(10)[[2, 3]]
        # >>> a
        # tensor([[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        # [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        # 水平方向最大的取值，忽略索引。意思是，除去真实标签，看看每个 batchsize 中哪个标签的概率最大，取出概率
        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
        # 选择真实标签的概率
        j = torch.masked_select(outputs, one_hot_labels.bool())

        # 如果有攻击目标，虚假概率减去真实概率，
        if targeted:
            return torch.clamp(i - j, min=-kappa)
        # 没有攻击目标，就让真实的概率小于虚假的概率，逐步降低，也就是最小化这个损失
        else:
            return torch.clamp(j - i, min=-kappa)

    w = torch.zeros_like(images, requires_grad=True).to(device)
    optimizer = optim.Adam([w], lr=learning_rate)
    #
    prev = 1e10

    for step in range(max_iter):
        a = 1 / 2 * (nn.Tanh()(w) + 1)
        # 第一个目标，对抗样本与原始样本足够接近
        loss1 = nn.MSELoss(reduction='sum')(a, images)
        # 第二个目标，误导模型输出
        loss2 = torch.sum(c * f(a))

        cost = loss1 + loss2
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter // 10) == 0:
            if cost > prev:
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost

    attack_images = 1 / 2 * (nn.Tanh()(w) + 1)

    return attack_images