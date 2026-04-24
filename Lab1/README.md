# AIH Lab1

## 简介

使用 Python + NumPy（支持 CuPy GPU 加速）从零构建深度学习框架，不依赖 PyTorch 等深度学习库，完成以下任务：

1. **反向传播算法**：实现全连接神经网络，拟合 sin(x) 回归 & 12 类手写汉字分类
2. **卷积神经网络**：手动实现 CNN 用于手写汉字分类

## 项目结构

```
Lab1/
├── module/          # 自研深度学习框架
│   ├── core.py          Module 基类（参数收集、梯度管理）
│   ├── layers.py        Linear, BatchNorm1d/2d, Dropout, MaxPool2d
│   ├── conv.py          Conv2d 卷积层
│   ├── activations.py   ReLU, Sigmoid, Tanh
│   ├── losses.py        MSE, MAE, CrossEntropy
│   ├── optimizers.py    SGD, SGDM, Adam
│   └── init.py          参数初始化
├── utils/           # 工具函数
│   ├── backend.py       NumPy/CuPy 自动切换
│   ├── data.py          数据加载
│   ├── plotting.py      可视化
│   └── training.py      训练循环
├── task1/            # Task1: sin(x) 回归拟合
├── task2/            # Task2: 手写汉字分类 (MLP & CNN)
└── data_2/           # 手写汉字数据集
```

## 快速开始

```bash
cd Lab1
pip install numpy cupy-cuda12x matplotlib   # cupy 按需安装，无 GPU 可仅用 numpy

# 1. 制作数据集（首次运行需要）
python -m task1.sine_dataset     # sin(x) 数据
python -m task2.image_dataset    # 手写汉字数据

# 2. Task1: sin(x) 回归
python -m task1.train
python -m task1.test

# 3. Task2: 手写汉字分类
python -m task2.train_mlp
python -m task2.train_cnn
python -m task2.test_mlp
python -m task2.test_cnn
```

## 环境依赖

- Python >= 3.8
- NumPy
- CuPy（可选，用于 GPU 加速）
- Matplotlib
