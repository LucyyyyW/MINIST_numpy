# MINIST_numpy
本实验我们基于numpy搭建双层全联接神经网络实现MINIST分类。

--main.py： 网络主体框架build_net(），有训练和测试过程代码。

--data.py:  读取MINIST数据集，以迭代器形式展现。

--tools.py(util.py): 模型存储，可视化代码

本实验网络结构为：两层全连接神经网络，layer1有1024神经元，layer2有10个神经元。

训练过程，batch_size：32，iteration num：12. 

代码使用方法：

For training:

"python main.py --train_img_path your_path --train_lab_path your_path --test_img_path your_path --test_lab_path your_path --resume your_path --state train"

For test:

"python main.py --test_img_path your_path --test_lab_path your_path --resume your_path --state train"
