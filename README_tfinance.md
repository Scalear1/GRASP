# TFinance OOD Detection

这个脚本专门用于在TFinance数据集上进行OOD（Out-of-Distribution）检测。

## 功能特点

- 支持TFinance数据集（通过DGL加载）
- 使用structure_shift方式创建OOD数据
- 支持多种OOD检测方法（GRASP, MSP, Energy, KNN等）
- 直接使用预训练的logits文件，无需重新训练模型
- 简洁的代码结构，易于使用和修改

## 使用方法

### 1. 准备数据

确保你的logits文件格式正确：
- 单个tensor：`logits.pt` (shape: [num_nodes, num_classes])
- 多个runs的list：`[logit1.pt, logit2.pt, ...]`

### 2. 运行单个实验

```bash
python test_ood_tfinance.py \
    --logits_path path/to/your/logits.pt \
    --ood GRASP \
    --device 0 \
    --runs 5 \
    --K 8 \
    --alpha 0.1 \
    --delta 0.1 \
    --tau1 10 \
    --tau2 50 \
    --st top \
    --ood_budget 0.1
```

### 3. 运行多个基线方法

```bash
# 修改run_tfinance_ood.sh中的LOGITS_PATH
chmod +x run_tfinance_ood.sh
./run_tfinance_ood.sh
```

## 参数说明

### 基本参数
- `--logits_path`: 预训练logits文件路径（必需）
- `--ood`: OOD检测方法（GRASP, MSP, Energy, KNN等）
- `--device`: GPU设备ID
- `--runs`: 实验运行次数
- `--ood_budget`: structure shift的扰动比例

### GRASP特定参数
- `--K`: 传播步数（默认8）
- `--alpha`: 传播权重（默认0.1）
- `--delta`: 增强系数（默认0.1）
- `--tau1`: 节点选择阈值（默认10）
- `--tau2`: 选择节点百分比（默认50）
- `--st`: 选择策略（top/low/random，默认top）

## 输出结果

脚本会输出：
- 每个run的AUROC和AUPR
- 平均性能和标准差
- 运行时间统计

## 示例输出

```
Final Results:
OOD Method: GRASP
Dataset: TFinance
Structure Shift Budget: 0.1
Average AUROC: 0.8234 ± 0.0123
Average AUPR: 0.7891 ± 0.0156
Average Time: 2.34s ± 0.45s
```

## 注意事项

1. 确保安装了所有依赖包（torch, torch_geometric, dgl等）
2. logits文件应该与数据集节点数量匹配
3. 如果使用CPU，添加`--cpu`参数
4. 可以根据需要调整structure_shift的budget参数