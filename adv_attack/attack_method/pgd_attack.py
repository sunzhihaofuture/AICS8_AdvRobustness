import torch, os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader


def pgd_attack(model, images, labels, eps, alpha=2/255, iters=40, device=None):
    images = images.to(device)
    labels = labels.to(device, dtype=torch.int64)
    loss = nn.CrossEntropyLoss()

    ori_images = images.data

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        # 通过 eps 限制扰动球面的范围
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach()

    return images
