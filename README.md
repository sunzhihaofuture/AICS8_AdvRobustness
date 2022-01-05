# AICS8_AdvRobustness


## 1. 将 cifar10 的训练集和测试集的图片保存下来，同时保存标签文件
修改save_path，即保存图片和标签文档的文件夹，part(test, train),分别用于保存训练集和测试集
```s
python data/cifar2image.py
```

## 2. 将图片生成 npy 的格式
修改 labelfile_path(这个是上面保存下来的标签文件)，code_dataset(保存结果名字)
```s
python train/gen_dataset.py
```

## 3. 练 resnet50 模型 和 densenet121 模型，在 cifar10 的测试集上训练
修改 code_experiment(实验名字，随便取）， code_dataset(上面保存npy的名字)
```s
python train/train.py
```

## 4. 生成攻击图片
修改 adv_attack/attack_main.py 文件中的 arch(resnet50, densenet121), code_experiment(自己随便取), code_attack_method(分别为cw，pgd，mifgsm), labelfile_path(攻击文件存放的地址), save_rootpath(生成的图片与标签存放的地址)

```s
python adv_attack/attack_main.py
```

注意：每次攻击图片只能生成1w张图片，所以 一个模型一种攻击方法想要生成2w张图片，需要跑两次

## 5. 手动选取5w张图片
从生成的所有图片文件中手动选取5w张图片，新建一个label_file

## 6. 将汇总的图片生成 npy 的格式
修改 labelfile_path(这个是上面保存下来的标签文件)，code_dataset(保存结果名字)
```s
python train/gen_dataset.py
```

## 7. 训练 resnet50 模型 和 densenet121 模型，在 cifar10 的测试集上训练
修改 code_experiment(实验名字，随便取）， code_dataset(上面保存npy的名字)
```s
python train/train.py
```


## TODO LIST

1. 5w张PGD攻击生成的图片训练(2.5w张resnet，2.5w张densenet) 提交
2. 5w张cw攻击生成的图片训练 提交
3. 5w张mifgsm攻击生成的图片训练 提交