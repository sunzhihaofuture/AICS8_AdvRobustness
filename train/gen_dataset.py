import numpy as np
import os
from PIL import Image
import torchvision
import random

labelfile_path = ''

image_path_list = []
image_label_list = []

code_dataset = 'FD2'

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


images = []
soft_labels = []

for i in range(len(image_path_list)):
    # print('{} {}'.format(image_path_list[i], image_label_list[i]))
    image_path = image_path_list[i]
    image_label = int(image_label_list[i])
    image = Image.open(image_path)
    image = np.asarray(image)
    images.append(image)
    soft_label = np.zeros(10)
    soft_label[image_label] += random.uniform(0, 10) # an unnormalized soft label vector
    soft_labels.append(soft_label)
images = np.array(images)
soft_labels = np.array(soft_labels)
print(images.shape, images.dtype, soft_labels.shape, soft_labels.dtype)
np.save('data/data-{}.npy'.format(code_dataset), images)
np.save('data/label-{}.npy'.format(code_dataset), soft_labels)