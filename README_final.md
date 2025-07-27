# TFinance OOD检测测试脚本

## 概述

我已经为你创建了一个完整的OOD检测测试框架，专门用于TFinance数据集。这个框架包含：

1. **logits生成脚本** - 生成测试用的logits文件
2. **OOD检测脚本** - 使用structure_shift方式进行OOD检测
3. **完整的测试流程** - 支持单次和多次运行

## 文件说明

### 1. 生成测试logits
- `generate_basic_logits.py` - 生成测试用的logits文件
- `test_logits_single.txt` - 单次运行的logits文件
- `test_logits_multiple.txt` - 多次运行的logits文件

### 2. OOD检测脚本
- `test_ood_final.py` - 主要的OOD检测脚本
- `test_simple.py` - 简化版测试脚本
- `test_minimal.py` - 最小化测试脚本

## 使用方法

### 1. 生成测试logits
```bash
python3 generate_basic_logits.py
```

### 2. 运行OOD检测
```bash
# 单次运行
python3 test_ood_final.py --logits_path test_logits_single.txt --runs 3 --ood_budget 0.1

# 多次运行
python3 test_ood_final.py --logits_path test_logits_multiple.txt --runs 5 --ood_budget 0.1
```

### 3. 参数说明
- `--logits_path`: logits文件路径
- `--runs`: 实验运行次数
- `--seed`: 随机种子
- `--ood_budget`: structure shift的扰动比例

## 功能特点

### 1. 数据加载
- 支持文本格式的logits文件
- 支持单次和多次运行格式
- 自动检测文件格式

### 2. Structure Shift OOD生成
- 基于社区结构的边扰动
- 可配置的扰动比例
- 保持图结构的完整性

### 3. 评估指标
- AUROC (Area Under ROC Curve)
- AUPR (Area Under Precision-Recall Curve)
- 支持多次运行的平均值和标准差

## 测试结果示例

```
==================================================
Final Results:
Dataset: TFinance (Mock)
Structure Shift Budget: 0.1
Average AUROC: 0.4956 ± 0.0020
Average AUPR: 0.8932 ± 0.0025
Average Time: 0.02s ± 0.00s
==================================================
```

## 文件格式

### 单次运行logits格式
```
10000 2
2.836561 1.449303
0.339264 1.060197
...
```

### 多次运行logits格式
```
5
10000 2
2.836561 1.449303
...
10000 2
1.234567 0.987654
...
```

## 注意事项

1. **环境要求**: 只需要Python标准库，无需额外依赖
2. **数据格式**: logits文件必须是文本格式
3. **节点数量**: 确保logits的节点数量与数据集匹配
4. **内存使用**: 对于大规模数据集，注意内存使用情况

## 扩展使用

你可以将这个框架扩展到：
1. 使用真实的TFinance数据集
2. 集成其他OOD检测方法
3. 添加更多的评估指标
4. 支持其他图数据集

这个框架提供了一个完整的OOD检测测试环境，你可以基于此进行进一步的研究和开发。