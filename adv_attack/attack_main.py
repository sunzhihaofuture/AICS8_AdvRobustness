import torch, os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

from torchattacks import *
from attack_method.cw_attack import cw_l2_attack
from attack_method.pgd_attack import pgd_attack

from model.resnet import ResNet50
from model.densenet import DenseNet121
from model.wideresnet import wideresnet

from utils.custom_datasets import Dataset_Cifar10
from utils.save_image import ImageSave


def main():
    # ---------- Set the Configuration ----------
    device = torch.device("cuda:0")
    arch = 'resnet50'
    code_experiment = ''
    code_save = ''
    code_attack_method = 'pgd'

    labelfile_path = ''
    image_rootpath = ''
    save_rootpath = ''

    if not os.path.exists(save_rootpath):
        os.makedirs(save_rootpath)

    # ---------- Load Model ----------
    print('Loading model {} of experiment {}...'.format(arch, code_experiment))

    if arch == 'resnet50':
        model = ResNet50()
        pth_file = 'path/resnet50-{}.pth.tar'.format(code_experiment)
    elif arch == 'densenet121':
        model = DenseNet121()
        pth_file = 'path/densenet121-{}.pth.tar'.format(code_experiment)
    elif arch == 'wideresnet':
        model = wideresnet()
        pth_file = 'path/wideresnet-{}.pth.tar'.format(code_experiment)

    net = torch.load(pth_file, map_location=device)['state_dict']
    model.load_state_dict({k.replace('module.',''):v for k, v in net.items()})
    model.to(device)

    # ---------- Load Dataset ----------
    print('Loading Dataset [{}]...'.format(labelfile_path))
    dataset = Dataset_Cifar10(labelfile_path=labelfile_path, image_rootpath=image_rootpath)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)

    # ---------- Attack and Save ----------
    print("Attacking with {}...".format(code_attack_method))
    save_data = None
    for data, target in tqdm(dataloader):
        data, target = data.float().to(device), target.long().to(device)
        
        if code_attack_method == 'cw':
            perturbed_data = cw_l2_attack(model=model, images=data, labels=target, device=device)
        elif code_attack_method == 'pgd':
            perturbed_data = pgd_attack(model=model, images=data, labels=target, eps=0.3)
        elif code_attack_method == 'mifgsm':
            attack_mifgsm = MIFGSM(model, eps=8/255, alpha=2/255, steps=100, decay=0.1)
            perturbed_data = attack_mifgsm(data, target)
        
        if save_data is None:
            save_data = perturbed_data.detach_().cpu().numpy()
        else:
            save_data = np.concatenate(
                (save_data, perturbed_data.detach_().cpu().numpy()), axis=0)

        saver = ImageSave()
        saver.save_images(save_data, labelfile_path, save_rootpath)


if __name__ == '__main__':
    main()
