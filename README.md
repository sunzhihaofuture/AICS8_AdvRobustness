# AICS8_AdvRobustness


## 将 cifar10 的训练集和测试集的图片保存下来，同时保存标签文件
修改save_path，即保存图片和标签文档的文件夹，part(test, train),分别用于保存训练集和测试集
python data/cifar2image.py

## 将图片生成 npy 的格式
修改 labelfile_path(这个是上面保存下来的标签文件)，code_dataset(保存结果名字)
python train/gen_dataset.py

## 训练 resnet50 模型 和 densenet121 模型，在 cifar10 的测试集上训练
修改 code_experiment(实验名字，随便取）， code_dataset(上面保存npy的名字)
python train/train.py

## 生成攻击图片
修改 adv_attack/attack_main.py 文件中的 arch(resnet50, densenet121), code_experiment(自己随便取), code_attack_method(分别为cw，pgd，mifgsm), labelfile_path(攻击文件存放的地址), save_rootpath(生成的图片与标签存放的地址)

```s
python adv_attack/attack_main.py
```
