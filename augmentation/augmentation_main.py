import numpy as np
import os
from PIL import Image
import imgaug.augmenters as iaa

labelfile_path = ''
image_rootpath = ''

augmentation1 = iaa.OneOf([
    iaa.Sequential([
        iaa.Fliplr(0.6),
        iaa.Flipud(0.6),
    ]),
    iaa.CoarseDropout((0.05, 0.15), size_percent=0.5),
    iaa.OneOf([
        iaa.GaussianBlur(sigma=(0.5, 2.0)),
        iaa.AverageBlur(k=(1, 5)),
        iaa.MedianBlur(k=(1, 5)),
        iaa.MotionBlur(k=3)
    ]),
    iaa.OneOf([
        iaa.MultiplyBrightness((0.8, 1.2)),
        iaa.MultiplyHue((0.8, 1.2)),
        iaa.MultiplySaturation((0.8, 1.2))
    ]), 
])

augmentation2 = iaa.SomeOf(2, [
    iaa.Sequential([
        iaa.Fliplr(0.6),
        iaa.Flipud(0.6),
    ]),
    iaa.CoarseDropout((0.05, 0.15), size_percent=0.5),
    iaa.OneOf([
        iaa.GaussianBlur(sigma=(0.5, 2.0)),
        iaa.AverageBlur(k=(1, 5)),
        iaa.MedianBlur(k=(1, 5)),
        iaa.MotionBlur(k=3)
    ]),
    iaa.OneOf([
        iaa.MultiplyBrightness((0.8, 1.2)),
        iaa.MultiplyHue((0.8, 1.2)),
        iaa.MultiplySaturation((0.8, 1.2))
    ]), 
])

def main():
    labelfile = open(labelfile_path, 'r')
    for line in labelfile:
        line = line[:-1]
        infos = line.split(' ')
        image_name = infos[0]
        image_label = infos[1]
            
        folder = os.path.join(image_rootpath, image_label)
        image_path = image_name
        
        image_saved_folder = folder.replace('cifar-10', 'cifar-10-augmentation1')
        image_saved_path = os.path.join(image_saved_folder, image_name)

        if os.path.exists(image_path):
            image_source = Image.open(image_path)
            image_source = np.asarray(image_source)
            image_target = augmentation1.augment_image(image_source)
            image_target = Image.fromarray(image_target)

            if not os.path.exists(image_saved_folder):
                os.makedirs(image_saved_folder)

            image_target.save(image_saved_path)

if __name__ == '__main__':
    main()