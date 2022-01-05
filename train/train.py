from __future__ import print_function

import os
import random
import shutil
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

from config.new_config import args_resnet, args_densenet, args_wideresnet

from utils import load_model, AverageMeter, accuracy

# Use CUDA
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:3')   # 用于选择cuda

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

code_experiment = 'cifar_test_attack_try_10000'   # 实验名称
code_dataset = 'cifar10_attack_try'   # 保存名字

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        images = np.load('data/data-{}.npy'.format(code_dataset))
        labels = np.load('data/label-{}.npy'.format(code_dataset))
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        assert images.shape[0] <= 50000
        assert images.shape[1:] == (32, 32, 3)
        self.images = [Image.fromarray(x) for x in images]
        self.labels = labels / labels.sum(axis=1, keepdims=True) # normalize
        self.labels = self.labels.astype(np.float32)
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.labels)

def cross_entropy(outputs, smooth_labels):
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    return loss(F.log_softmax(outputs, dim=1), smooth_labels)

def main():
    for arch in ['densenet121', 'resnet50', 'wideresnet']:
        if arch == 'resnet50':
            args = args_resnet
        elif arch == 'densenet121':
            args = args_densenet
        elif arch == 'wideresnet':
            args = args_wideresnet
        assert args['epochs'] <= 200

        # Data
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = MyDataset(transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=4)
        
        # Model
        best_acc = 0  # best test accuracy

        model = load_model(arch)

        optimizer = optim.__dict__[args['optimizer_name']](model.parameters(),
            **args['optimizer_hyperparameters'])
        if args['scheduler_name'] != None:
            scheduler = torch.optim.lr_scheduler.__dict__[args['scheduler_name']](optimizer,
            **args['scheduler_hyperparameters'])
        model = model.to(device)

        # Train and val
        for epoch in tqdm(range(args['epochs'])):
            train_loss, train_acc = train(trainloader, model, optimizer, device)
            print(args)
            print('acc: {}'.format(train_acc))

            # save model
            best_acc = max(train_acc, best_acc)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': train_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, arch=arch)
            if args['scheduler_name'] != None:
                scheduler.step()
            
            # write log
            logfile = open('logfile/{}-{}.txt'.format(arch, code_experiment), 'a')
            logfile.write('epoch:[{}|{}], acc:{}, best_acc:{}, loss:{}\n'.format(epoch, args['epochs'], train_acc, best_acc, train_loss))
            logfile.close()

        print('Best acc:')
        print(best_acc)


def train(trainloader, model, optimizer, device):
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()
    # switch to train mode
    model.train()

    for (inputs, soft_labels) in trainloader:
        inputs, soft_labels = inputs.to(device), soft_labels.to(device)
        targets = soft_labels.argmax(dim=1)
        outputs = model(inputs)
        loss = cross_entropy(outputs, soft_labels)
        acc = accuracy(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        accs.update(acc[0].item(), inputs.size(0))
    return losses.avg, accs.avg

def save_checkpoint(state, arch):
    filepath = os.path.join('output/{}-{}.pth.tar'.format(arch, code_experiment))
    torch.save(state, filepath)

if __name__ == '__main__':
    main()
