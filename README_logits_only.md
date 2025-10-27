# 使用预计算Logits运行GRASP OOD检测

## 概述

现在您可以**完全跳过模型训练和推理**，直接使用预计算的logits进行OOD检测。这样做的好处：

- ✅ **无需GPU训练**：只需要预计算的logits文件
- ✅ **快速实验**：直接测试不同的OOD检测方法
- ✅ **可重复性**：使用相同的logits确保结果一致
- ✅ **节省时间**：跳过耗时的模型推理过程

## 文件说明

- `test_ood_logits_only.py` - **推荐使用**：专门为预计算logits设计的简化版本
- `test_ood.py` - 原版本，已修改支持预计算logits，但保留了模型相关代码

## Logits文件格式要求

### 1. 数据格式
```python
# logits应该是一个torch.Tensor
# 形状: [num_nodes, num_classes]
# 数据类型: torch.float32
# 示例:
logits = torch.randn(1000, 7)  # 1000个节点，7个类别
torch.save(logits, 'your_logits.pt')
```

### 2. 文件组织方式

#### 方式1: 单个logits文件（推荐）
```
logits/
└── your_dataset_logits.pt  # 包含所有节点的logits
```

#### 方式2: 按run分别存储
```
logits/
└── your_dataset/
    └── gcn/
        ├── logit0.pt  # run 0的logits
        ├── logit1.pt  # run 1的logits
        ├── logit2.pt  # run 2的logits
        └── ...
```

## 使用方法

### 1. 基本使用

```bash
# 使用单个logits文件
python test_ood_logits_only.py \
    --dataset your_dataset_name \
    --method gcn \
    --ood GRASP \
    --logits_file logits/your_dataset_logits.pt \
    --force_logits \
    --runs 5 \
    --device 0

# 使用多个logits文件
python test_ood_logits_only.py \
    --dataset your_dataset_name \
    --method gcn \
    --ood GRASP \
    --logits_dir logits/your_dataset/gcn \
    --force_logits \
    --runs 5 \
    --device 0
```

### 2. 支持的OOD检测方法

```bash
# GRASP方法（推荐作为baseline）
python test_ood_logits_only.py --dataset your_dataset --ood GRASP --logits_file your_logits.pt

# MSP方法
python test_ood_logits_only.py --dataset your_dataset --ood MSP --logits_file your_logits.pt

# Energy方法
python test_ood_logits_only.py --dataset your_dataset --ood Energy --logits_file your_logits.pt

# KNN方法
python test_ood_logits_only.py --dataset your_dataset --ood KNN --logits_file your_logits.pt

# ODIN方法
python test_ood_logits_only.py --dataset your_dataset --ood ODIN --logits_file your_logits.pt

# Mahalanobis方法
python test_ood_logits_only.py --dataset your_dataset --ood Mahalanobis --logits_file your_logits.pt
```

### 3. 批量运行脚本

使用提供的 `run_with_precomputed_logits.py`:

```bash
# 1. 修改脚本中的数据集名称
# 2. 确保logits文件存在
# 3. 运行脚本
python run_with_precomputed_logits.py
```

## 参数说明

### 必需参数
- `--dataset`: 数据集名称
- `--ood`: OOD检测方法
- `--logits_file` 或 `--logits_dir`: logits文件路径

### 重要可选参数
- `--force_logits`: 强制使用logits模式，找不到文件时退出
- `--runs`: 运行次数（默认5）
- `--device`: GPU设备号（默认0）
- `--train_prop`: 训练集比例（默认0.1）
- `--valid_prop`: 验证集比例（默认0.1）

## 准备您的Logits文件

### 从您的模型生成logits

```python
import torch
from your_model import YourModel  # 您的模型
from your_data_loader import load_your_data  # 您的数据加载器

# 1. 加载训练好的模型
model = YourModel()
model.load_state_dict(torch.load('your_model.pth'))
model.eval()

# 2. 加载数据
graph_data = load_your_data()

# 3. 生成logits
with torch.no_grad():
    logits = model(graph_data)  # 形状: [num_nodes, num_classes]

# 4. 保存logits
torch.save(logits, 'logits/your_dataset_logits.pt')
print(f"Logits saved! Shape: {logits.shape}")
```

### 验证logits格式

```python
import torch

# 加载并检查logits
logits = torch.load('your_logits.pt')
print(f"Logits shape: {logits.shape}")
print(f"Logits dtype: {logits.dtype}")
print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")

# 检查是否包含NaN或Inf
print(f"Contains NaN: {torch.isnan(logits).any()}")
print(f"Contains Inf: {torch.isinf(logits).any()}")
```

## 预期输出

成功运行后，您会看到类似输出：

```
=== OOD Detection with Precomputed Logits ===
Dataset: your_dataset_name
OOD Method: GRASP
Runs: 5
Device: cuda:0

Dataset info: 2708 nodes | 7 classes | 1433 features
Structure OOD detected: ID test nodes 271, OOD nodes 271
Using single logits file: logits/your_dataset_logits.pt

--- Run 1/5 ---
✓ Loading logits from: logits/your_dataset_logits.pt
  Logits shape: torch.Size([2708, 7])
  Applying GRASP method...
  Results - AUROC: 0.8532, AUPR: 0.7845, FPR95: 0.1234

--- Run 2/5 ---
...

=== Final Results ===
Method: GRASP
Successful runs: 5/5
Average time per run: 2.3456s

Results:
AUROC: 85.32 ± 2.15
AUPR: 78.45 ± 3.21
FPR95: 12.34 ± 1.87

✅ Completed successfully!
```

## 故障排除

### 1. Logits文件找不到
```bash
❌ Logits file not found: logits/your_dataset_logits.pt
```
**解决方案**: 检查文件路径是否正确，确保logits文件存在

### 2. Logits维度不匹配
```bash
❌ Logits shape mismatch! Expected (2708, 7), got (2708, 10)
```
**解决方案**: 检查您的logits是否对应正确的数据集和类别数

### 3. 内存不足
**解决方案**: 使用CPU模式 `--cpu` 或减少batch size

## 优势总结

1. **简化流程**: 无需处理模型加载、权重管理等复杂问题
2. **快速实验**: 直接测试各种OOD检测方法的效果
3. **一致性**: 所有方法使用相同的logits，确保公平比较
4. **灵活性**: 可以轻松测试不同的超参数配置
5. **可重复**: 完全确定性的结果，便于论文实验

现在您可以专注于OOD检测方法的比较，而不用担心模型训练的复杂性！