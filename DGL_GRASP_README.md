# 在DGL数据集上使用GRASP进行OOD检测

本文档说明如何修改GRASP代码以支持DGL格式的图数据集，并使用预训练GNN模型的logits进行Out-of-Distribution (OOD) 检测。

## 修改的文件

1. **`load_dgl_data.py`** - 新增：DGL数据加载和转换函数
2. **`dataset.py`** - 修改：添加对DGL数据集的支持
3. **`parse.py`** - 修改：添加DGL相关的命令行参数
4. **`test_dgl_dataset.py`** - 新增：专门用于测试DGL数据集的脚本
5. **`example_usage.py`** - 新增：使用示例和说明

## 前提条件

确保你有以下数据：
- DGL格式的图数据文件
- 预训练GNN模型产生的logits结果

### DGL数据格式要求

你的DGL图数据应该包含：
```python
graph = load_graphs(prefix + name)[0][0]
node_feats = graph.ndata['feature']  # 节点特征
node_labels = graph.ndata['label']   # 节点标签  
edge_index = graph.edges()           # 边连接
```

### Logits格式要求

预训练模型的logits应该是：
- 类型：`torch.Tensor`
- 形状：`[num_nodes, num_classes]`
- 对应于图中每个节点的分类logits

## 使用方法

### 1. 准备数据

首先，确保你的预训练logits已保存为PyTorch tensor：

```python
import torch

# 假设你已经有预训练模型的logits
# logits = your_pretrained_model(graph_data)  # 形状: [num_nodes, num_classes]

# 保存logits
torch.save(logits, 'your_logits.pt')
```

### 2. 运行GRASP测试

使用提供的测试脚本：

```bash
python test_dgl_dataset.py \
    --data_path /path/to/your/data/ \
    --dataset_name your_dataset.bin \
    --logits_path your_logits.pt \
    --create_ood_split \
    --ood_ratio 0.2 \
    --runs 5 \
    --alpha 0.1 \
    --K 5
```

### 3. 参数说明

#### 必需参数：
- `--data_path`: DGL数据文件的路径前缀
- `--dataset_name`: DGL数据集文件名
- `--logits_path`: 预训练模型logits文件路径

#### OOD分割参数：
- `--create_ood_split`: 是否创建OOD分割
- `--ood_ratio`: OOD数据的比例 (默认0.2)
- `--ood_type`: OOD分割类型 ('random' 或 'label_based')

#### GRASP参数：
- `--alpha`: 信息传播权重 (默认0.1)
- `--K`: 传播步数 (默认5)
- `--delta`: 增强节点权重 (默认1.0)
- `--tau1`: 节点选择阈值1 (默认20.0)
- `--tau2`: 节点选择阈值2 (默认50.0)
- `--st`: 节点选择策略 ('top', 'low', 'random', 'test')

#### 其他参数：
- `--runs`: 运行次数 (默认5)
- `--seed`: 随机种子 (默认1)
- `--device`: GPU设备号 (默认0)
- `--cpu`: 使用CPU运行

## 示例使用

查看 `example_usage.py` 获取完整的使用示例：

```bash
python example_usage.py
```

## 工作流程

1. **数据加载**：`load_dgl_data.py` 中的函数将DGL格式转换为torch_geometric.data.Data格式
2. **OOD分割**：如果指定，会创建ID和OOD节点的分割
3. **GRASP检测**：使用预训练logits和图结构进行OOD检测
4. **结果评估**：计算AUROC、AUPR和FPR95指标

## 输出结果

测试完成后，结果会保存到 `results/custom-dgl-GRASP.csv`，包含：
- AUROC (Area Under ROC Curve)
- AUPR (Area Under Precision-Recall Curve)  
- FPR95 (False Positive Rate at 95% True Positive Rate)

## 注意事项

1. **内存使用**：大型图可能需要大量内存，考虑使用 `--cpu` 参数或减少batch size
2. **参数调优**：GRASP的性能很大程度上依赖于参数设置，建议根据你的数据集调优
3. **OOD定义**：确保OOD分割符合你的实际应用场景
4. **logits质量**：GRASP的效果依赖于预训练模型logits的质量

## 故障排除

### 常见错误：

1. **形状不匹配**：确保logits的节点数与图的节点数一致
2. **内存不足**：尝试使用CPU或减小图的规模
3. **DGL版本**：确保安装了兼容的DGL版本

### 调试建议：

1. 首先在小规模数据上测试
2. 检查数据加载是否正确
3. 验证logits的形状和数值范围
4. 逐步调整GRASP参数

## 扩展功能

你可以根据需要扩展以下功能：

1. **自定义OOD分割策略**：修改 `create_custom_ood_split()` 函数
2. **不同的评估指标**：添加其他OOD检测指标
3. **批处理支持**：对于超大图的批处理处理
4. **可视化**：添加结果可视化功能

## 联系和支持

如果遇到问题，请检查：
1. 数据格式是否正确
2. 参数设置是否合理
3. 依赖包是否安装完整

祝你使用愉快！