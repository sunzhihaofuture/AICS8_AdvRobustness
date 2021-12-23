from models import *
import os
from PIL import Image
import time
import torch
from torchvision import transforms

def load_model(arch):
    model = globals()[arch]()
    model.eval()
    return model

def main():
    code_experiment = ''

    device = torch.device('cuda:0')

    nets_path = {'resnet50': 'path/resnet50-{}.pth.tar'.format(code_experiment),
                 'densenet121': 'path/densenet121-{}.pth.tar'.format(code_experiment)}
    
    labelfile_path = ''
    hard_sample_labelfile_path = ''

    image_path_list = []
    image_label_list = []

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
    
    hard_sample_infos_list = []  # 将当前所有模型的难样本统计出来，不区分模型，注意去重

    for arch in ['resnet50', 'densenet121']:
        model = load_model(arch)
        net = torch.load(nets_path[arch], map_location=device)['state_dict']
        model.load_state_dict({k.replace('module.',''):v for k, v in net.items()})
        model.to(device)

        correct = 0
        for i in range(len(image_path_list)):
            image_path = image_path_list[i]
            image_label = image_label_list[i]
            image_pil = Image.open(image_path)
            
            data_transform = transforms.Compose(
                [transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            image_pil = data_transform(image_pil)
            image_pil = torch.unsqueeze(image_pil, dim=0)

            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(image_pil.to(device)))
                predict = torch.softmax(output, dim=0).cpu()
                predict_class = torch.argmax(predict).numpy()
                proba = predict[predict_class].numpy()

            print(image_path, proba, predict_class, image_label)
            
            if predict_class == int(image_label):
                correct += 1
            else:
                hard_sample_infos_list.append('{} {}'.format(image_path, image_label))
                
            acc = correct / (i + 1) * 100
            print('sample:[{}|{}], acc: {:2f}'.format(i+1, len(image_path_list), acc))

            logfile = open('logfile/{}_inference.txt'.format(arch), 'w')
            logfile.write('sample:[{}|{}], acc: {:2f}\n'.format(i+1, len(image_path_list), acc))
            logfile.close()

    # 去重
    hard_sample_infos_list = list(set(hard_sample_infos_list))

    # 写入最终的标签文件
    labelfile = open(hard_sample_labelfile_path, 'w')
    for index in range(len(hard_sample_infos_list)):
        labelfile.write('{}\n'.format(hard_sample_infos_list[index]))
    labelfile.close()


if __name__ == '__main__':
    main()