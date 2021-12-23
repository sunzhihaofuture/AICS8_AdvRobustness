import numpy as np
import os
from PIL import Image


def np2image(image_np, image_name, image_save_folder):
    # shape of image_np input is float, but PIL.Image.save() need uint8
    image_np = image_np.astype(np.uint8)
    image_pil = Image.fromarray(image_np)

    if not os.path.exists(image_save_folder):
        os.makedirs(image_save_folder)
    image_pil.save('{}/{}'.format(image_save_folder, image_name))
    print('{} has been saved into {}.'.format(image_name, image_save_folder))

def main():
    labelfile_path = ''
    image_rootpath = ''

    image1_path = os.path.join(image_rootpath, '7/broodmare_s_000739.png')
    image1 = Image.open(image1_path)
    image1 = np.asarray(image1)
    label1 = 7
    soft_label1 = np.zeros(10)
    soft_label1[label1] += 1

    image2_path = os.path.join(image_rootpath, '1/police_cruiser_s_000503.png')
    image2 = Image.open(image2_path)
    image2 = np.asarray(image2)
    label2 = 1
    soft_label2 = np.zeros(10)
    soft_label2[label2] += 1

    alpha = 0.4
    lam = np.random.beta(alpha, alpha)
    print(lam)
    image = lam * image1 + (1 - lam) * image2
    label = lam * soft_label1 + (1 - lam) * soft_label2
    print(image)
    print(label)
    np2image(image, 'example.png', 'example')

if __name__ == '__main__':
    main()