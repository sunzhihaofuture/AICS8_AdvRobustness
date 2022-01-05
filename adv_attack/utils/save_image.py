import numpy as np
import os, cv2
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
    
    def save_images(self, image_list_np, image_rootpath):

        for index in range(image_list_np.shape[0]):
            name = '0'*(7-len(str(index))) + str(index) + '.jpg'
            image_path = f'{image_rootpath}/images/{name}'
            self.save_image(image_list_np[index], image_path)
