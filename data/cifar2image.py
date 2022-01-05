import numpy as np
import torchvision
import random, cv2, os

if __name__ == "__main__":
    for part in ['train', 'test']:
        save_path = f"/home/linyan/project/Tianchi/AICS8_AdvRobustness/data/images/{part}"  # images
        dataset = torchvision.datasets.CIFAR10(root='data/', train=False if part == 'test' else True, download=False)
        images = []
        labels = []
        for image, label in dataset:
            image = np.array(image)
            images.append(image)

            labels.append(label)

        f = open(f'data/cifar10_{part}.txt', 'w')
        for idx, (image, label) in enumerate(zip(images, labels)):
            name = '0'*(7-len(str(idx))) + str(idx) + '.jpg'
            cv2.imwrite(os.path.join(save_path, name), image)
            f.write(os.path.join(save_path, name) + ' ' + str(label))
            f.write('\n')

        f.close()
