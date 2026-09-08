# 复旦大学 人工智能(H) 课程项目

Lab1 使用 Python + NumPy（支持 CuPy GPU 加速）从零构建深度学习框架，不依赖 PyTorch 等深度学习库，完成以下任务：

1. **反向传播算法**：实现全连接神经网络，拟合 sin(x) 回归 & 12 类手写汉字分类
2. **卷积神经网络**：手动实现 CNN 用于手写汉字分类

Lab2 命名实体识别（NER）实验，使用四种模型完成中英文 NER 任务：

1. **HMM**：从零实现隐马尔可夫模型，Viterbi 解码
2. **CRF**：基于 PyTorch 实现 Linear-CRF（Embedding + Linear + CRF 层）
3. **Transformer**：BERT 特征提取 + CRF Viterbi 解码
4. **Bonus**：基于 CRF++ 模板的显式特征 CRF，严格区分 Unigram/Bigram