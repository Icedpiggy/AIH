# AIH Lab2

## 简介

命名实体识别（NER）实验，使用四种模型完成中英文 NER 任务：

1. **HMM**：从零实现隐马尔可夫模型，Viterbi 解码
2. **CRF**：基于 PyTorch 实现 Linear-CRF（Embedding + Linear + CRF 层）
3. **Transformer**：BERT 特征提取 + CRF Viterbi 解码
4. **Bonus**：基于 CRF++ 模板的显式特征 CRF，严格区分 Unigram/Bigram

## 项目结构

```
Lab2/
├── utils/            # 共享工具
│   ├── data.py           # load_data, build_vocab, tag mappings
│   └── crf.py            # CRF 层（前向算法、Viterbi、负对数似然）
├── task1/            # HMM
│   ├── data_utils.py     # 数据加载、标签映射
│   ├── HMM.py            # HMM 模型类
│   ├── train.py          # 训练入口
│   └── test.py           # 预测 + 评测
├── task2/            # CRF
│   ├── model.py          # LinearCRF 类
│   ├── data.py           # NERDataset + collate_fn
│   ├── train.py          # 训练入口
│   └── test.py           # 预测 + 评测
├── task3/            # Transformer
│   ├── model.py          # TransformerNER 类（BERT + Linear + CRF）
│   ├── data.py           # NERDataset（BERT tokenizer + 子词对齐）
│   ├── train.py          # 训练入口
│   └── test.py           # 预测 + 评测
├── extra/            # Bonus: 模板驱动 CRF
│   ├── template.py       # 模板解析 + 特征提取
│   ├── model.py          # TemplateCRF 类（动态转移矩阵）
│   ├── data.py           # 数据加载
│   ├── train.py          # 训练入口
│   └── test.py           # 预测 + 评测
├── NER/              # 数据与评测
└── report.pdf            # 实验报告
```

## 快速开始

```bash
cd Lab2
pip install torch transformers scikit-learn

# HMM
python -m task1.train Chinese 0 && python -m task1.test Chinese
python -m task1.train English 0 && python -m task1.test English

# CRF
python -m task2.train Chinese && python -m task2.test Chinese
python -m task2.train English && python -m task2.test English

# Transformer
python -m task3.train Chinese && python -m task3.test Chinese
python -m task3.train English && python -m task3.test English

# Bonus
python -m extra.train Chinese && python -m extra.test Chinese
python -m extra.train English && python -m extra.test English
```

## 最终结果

| 模型 | Chinese F1 | English F1 |
|------|-----------|------------|
| HMM | 0.9009 | 0.7745 |
| CRF | 0.9151 | 0.8227 |
| Transformer | 0.9734 | 0.9519 |
| Template CRF | 0.9413 | 0.7195 |

## 环境依赖

- Python >= 3.8
- PyTorch
- Transformers (HuggingFace)
- scikit-learn
- NumPy
