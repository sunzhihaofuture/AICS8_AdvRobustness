import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import os


class AddSaltPepperNoise(object):
    def __init__(self, density=0):
        self.density = density
    
    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[Nd/2.0, Nd/2.0, Sd])
        mask = np.repeat(mask, c, axis=2)
        img[mask == 0] = 0
        img[mask == 1] = 255
        img= Image.fromarray(img.astype('uint8')).convert('RGB')
        return img
    
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=20.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
    
    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img

if __name__ == '__main__':
    my_transform_gaussian_noise = AddGaussianNoise()
    my_transform_saltpepper_noise = AddSaltPepperNoise()
    trans = transforms.RandomChoice([
        my_transform_saltpepper_noise,
        my_transform_gaussian_noise,
    ])

    labelfile_path = 'data/cifar10_test.txt'
    save_img_path = 'data/aug_noise/image'
    save_path = 'data/aug_noise'
    idx = 0
    imgs_file_list = []
    
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path, exist_ok=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    with open(labelfile_path, 'r') as labelfile:
        for line in labelfile:
            line = line[:-1]
            infos = line.split(' ')
            image_name = infos[0]
            image_label = infos[1]
            
            image_path = image_name

            if os.path.exists(image_path):
                img = Image.open(image_path)
                name = '0'*(7-len(str(idx))) + str(idx) + '.jpg'
                save = os.path.join(save_img_path, name)
                imgs_file_list.append(f'{save} {image_label}')
                img.save(save)
                idx += 1
                for i in range(4):
                    img = trans(img)
                    name = '0'*(7-len(str(idx))) + str(idx) + '.jpg'
                    save = os.path.join(save_img_path, name)
                    imgs_file_list.append(f'{save} {image_label}')
                    img.save(save)
                    idx += 1
    
    f = open(f'{save_path}/attack.txt', 'w')
    for idx, content in enumerate(imgs_file_list):
        f.write(content)
        f.write('\n')

            
