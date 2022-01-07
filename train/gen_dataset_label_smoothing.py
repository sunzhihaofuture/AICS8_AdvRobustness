import os
import torchvision
import random
import numpy as np
from PIL import Image
import cv2
from albumentations import (
    IAAPerspective, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, PiecewiseAffine,
    Sharpen, Emboss, RandomBrightnessContrast, OneOf, Compose, Cutout, CoarseDropout, ShiftScaleRotate,
)
import torchvision
from tqdm.std import trange
from tqdm import tqdm
def get_train_transforms():
    return Compose(
        [
            Transpose(p=0.25),
            GaussNoise(p=0.75),
            OneOf([
                    # 模糊相关操作
                    MotionBlur(p=.75),
                    MedianBlur(blur_limit=3, p=0.5),
                    Blur(blur_limit=3, p=0.75),
                ], p=0.25),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.25),
            OneOf([
                # 畸变相关操作
                OpticalDistortion(p=0.75),
                GridDistortion(p=0.25),
                PiecewiseAffine(p=0.75),
            ], p=0.25),
            OneOf([
                    # 锐化、浮雕等操作
                    CLAHE(clip_limit=2),
                    Sharpen(),
                    Emboss(),
                    RandomBrightnessContrast(),            
                ], p=0.25),
            #
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            OneOf(
                [
                CoarseDropout(max_holes=4,
                            max_height=4,
                            max_width=4,
                            p=0.5),
                Cutout(
                    num_holes=4,
                    max_h_size=4,
                    max_w_size=4,
                    p=0.5,)],
                p=0.5)
        ]
        )


confidence = 0.8
smoothing = 0.2
cls= 10 
dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False)
images = []
soft_labels = []
transform=get_train_transforms()
cnt=-1

labelfile_path = 'data/attack/cifar10_attack.txt'

image_path_list = []
image_label_list = []

code_dataset = 'cifar10_attack'

with open(labelfile_path, 'r') as labelfile:
    for line in labelfile:
        line = line[:-1]
        infos = line.split(' ')
        image_name = infos[0]
        image_label = infos[1]
        
        image_path = image_name

        if os.path.exists(image_path):
            image_path_list.append(image_path)
            image_label_list.append(image_label)


for i in range(len(image_path_list)):
    # print('{} {}'.format(image_path_list[i], image_label_list[i]))
    image_path = image_path_list[i]
    image_label = int(image_label_list[i])
    image = Image.open(image_path)
    image = np.asarray(image)
    images.append(image)
    soft_label = np.zeros(10)
    soft_label[image_label] += confidence # an unnormalized soft label vector
    for inx in range(soft_label.shape[0]):
        if inx!=image_label:
            soft_label[inx]=smoothing / (cls - 1)
    soft_labels.append(soft_label)

for image, label in tqdm(dataset):
    image = np.array(image)
    #images.append(image)
    soft_label = np.zeros(10)
    soft_label[label] += confidence # an unnormalized soft label vector
    for inx in range(soft_label.shape[0]):
        if inx!=label:
            soft_label[inx]=smoothing / (cls - 1)
    #soft_labels.append(soft_label)
    #-----------image aug------------
    #cnt+=1
    #if cnt<10000:
    #数据增强
    for _ in range(0):
        img=transform(image=image)['image']
        images.append(img)
        soft_labels.append(soft_label)

images = np.array(images)
soft_labels = np.array(soft_labels)
print(images.shape, images.dtype, soft_labels.shape, soft_labels.dtype)
np.save('data/data-{}.npy'.format(code_dataset), images)
np.save('data/label-{}.npy'.format(code_dataset), soft_labels)