# 项目代码说明
本项目为: 利用多层感知机（Multilayer Perceptron, MLP）预测未知结构参数的卷式膜进水通道流场速度、压力、浓度等物理性质分布.
此项目中MLP为纯数据驱动模型, 利用MLP学习650组设计参数下的通道流场数据，预测出新设计参数下的速度、压力、浓度等.
经过与高保真三维CFD方式拟合出的流场对比，可以发现MLP模型对流场的物理性质分布较为准确, 有较好的应用价值.

# 代码目录结构
```
    code/
    ├─.git/
    │ 
    ├─compare_pictures/
    │  ├─pictures_compare_functions/
    │  └─results/
    ├─csv_datasets/
    ├─data_preprocessing/
    │  └─not usual/
    ├─experiment_record/
    ├─models/
    ├─model_training/
    │  ├─mlp_training_all/
    │  ├─mlp_training_inside/
    │  ├─mlp_training_outside/
    │  └─old_mlp_training/
    ├─predict/
    │  ├─old_predict/
    │  ├─predict_all/
    │  ├─predict_functions/
    │  ├─predict_inside/
    │  └─predict_outside/
    ├─results/
    │  ├─data_processing_log/
    │  ├─model_parameters/
    │  ├─predict_processing_log/
    │  ├─training_process_output/
    │  └─training_process_pictures/
    ├─south_2/
    │  ├─predict/
    │  └─train/
    └─utility/
```


# 代码目录说明
./.git/ 为 git 文件，无需点开，用于记录版本变化
./compare_pictures/ 为图像相似度比较的代码
./csv_datasets/ 为速度大小、压力、浓度的训练集和验证集类库
./data_preprocessing/ 为数据前处理、后处理代码
./experiment_record/ 为所有训练过程的实验计划
./models/ 为用到的神经网络模型类库
./model_training/ 为神经网络训练代码
./predict/ 为 MLP 预测流场速度大小、压力、浓度的代码
./results/ 为所有训练、预测、数据处理过程的实验记录，以及训练得到的 MLP 模型参数合集
./south_2/ 为在南方二号上进行 MLP 模型训练及预测的代码
./utility/ 为实现某一特定功能的代码


# 环境依赖
本项目的 python3 代码运行需要安装的第三方包写在./requirements.txt文件里，运行代码前需要在工作的 python 环境中运行```pip3 install -r requirements.txt```.
# 注意事项
1.请勿改动代码目录结构和名称，否则代码也需要做出相应的改变
2.由于数据文件太大，../data/ 文件夹没有一起放到代码文件下面，而是与../code/文件夹放在同一层级下
3.由于本工作代码的设置，在Windows系统、MacOS系统以及Linux系统下都可以运行本项目的代码
4.运行代码时，无需设置特定的工作目录，设置工作目录的步骤已经写进 python3 代码了，为了使用者方便，可以将工作目录设置为 code/ 目录
5.本项目代码在 python3.7 中运行， 在 python3.9(包括3.9)的版本下运行良好，更高的 python3 版本则没有测试过
6.本项目代码在 GPU 或 CPU 环境下都能运行，推荐在GPU环境下训练 MLP 神经网络模型

