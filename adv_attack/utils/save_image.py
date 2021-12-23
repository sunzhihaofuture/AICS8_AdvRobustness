import numpy as np
import os
from PIL import Image

class ImageSave():
    def __init__(self):
        pass

    def save_image(self, image_np, save_path):
        image_name = save_path.split('/')[-1]
        save_folder = save_path.replace(image_name, '')

        # shape of image_np input is (3, 32, 32) and float, but PIL.Image.save() need (32, 32, 3) and uint8
        image_np = image_np.transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)
        image = Image.fromarray(image_np)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        image.save(save_path)
        print('{} has been saved into {}.'.format(image_name, save_folder))
    
    def save_images(self, image_list_np, labelfile_path, image_rootpath):
        image_path_list = []
        
        labelfile = open(labelfile_path, 'r')
        for line in labelfile:
            line = line[:-1]
            infos = line.split(' ')
            image_name = infos[0]
            image_label = infos[1]
                
            folder = '{}/{}'.format(image_rootpath, image_label)
            image_path = '{}/{}'.format(folder, image_name)
            image_path_list.append(image_path)

        # assert image_list_np.shape[0] == len(image_path_list)

        for index in range(image_list_np.shape[0]):
            self.save_image(image_list_np[index], image_path_list[index])
