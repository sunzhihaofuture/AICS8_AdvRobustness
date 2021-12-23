import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


def compute_difference(image_ori, image_adv):
    image_diff_np = np.asarray(image_adv) - np.asarray(image_ori)
    image_diff = Image.fromarray(image_diff_np)
    image_diff = image_diff.convert('L')
    return image_diff

def show_image_contrast(image_ori, image_adv, title):
    plt.figure(figsize=(10, 4))     
    plt.suptitle(title)

    plt.subplot(1, 3, 1), 
    plt.title('Origin Image')
    plt.imshow(image_ori)
    plt.axis('off')

    plt.subplot(1, 3, 2), 
    plt.title('Adversarial Attack Image')
    plt.imshow(image_adv)
    plt.axis('off')

    plt.subplot(1, 3, 3), 
    plt.title('Contrast in Gray')
    plt.imshow(compute_difference(image_ori, image_adv), cmap='gray')
    plt.axis('off')

    # plt.show()
    plt.savefig('adv_image')

def main():
    path1 = 'path/coupe_s_000629.png'
    path2 = path1.replace('path/image_ori', 'path/image_ori')

    im1 = Image.open(path1)
    im2 = Image.open(path2)
    
    show_image_contrast(im1, im2, 'test')


if __name__ == '__main__':
    main()