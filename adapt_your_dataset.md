# 如何适配您的DGL数据集到GRASP框架

## 1. 修改数据集加载函数

### 步骤1：更新 `dataset.py` 中的 `load_your_dgl_dataset` 函数

```python
def load_your_dgl_dataset(args):
    """
    加载您的DGL格式数据集并生成structure OOD数据
    """
    from dgl import load_graphs
    import torch
    from torch_geometric.data import Data
    
    # 🔧 请根据您的实际情况修改以下参数
    prefix = "/path/to/your/dataset/"  # 您的数据集路径
    name = f"{args.dataset}.bin"       # 您的数据集文件名格式
    
    # 加载DGL图数据
    graph = load_graphs(prefix + name)[0][0] 
    node_feats = graph.ndata['feature']  # 🔧 如果特征字段名不同，请修改
    node_labels = graph.ndata['label']   # 🔧 如果标签字段名不同，请修改
    edge_index = graph.edges()
    
    # 转换为PyTorch Geometric格式
    edge_index = torch.stack([edge_index[0], edge_index[1]], dim=0)
    
    # 🔧 数据类型转换（根据需要）
    if node_feats.dtype != torch.float32:
        node_feats = node_feats.float()
    if node_labels.dtype != torch.long:
        node_labels = node_labels.long()
    
    # 创建PyTorch Geometric Data对象
    data = Data(
        x=node_feats,
        edge_index=edge_index,
        y=node_labels,
        num_nodes=node_feats.shape[0]
    )
    
    # 添加必要的属性
    data.node_idx = torch.arange(data.num_nodes)
    
    # 生成结构偏移的OOD数据
    run = getattr(args, 'run', 0)
    dataset_train_id, dataset_test_id, dataset_ood = structure_shift_dataset(
        data, run, args, ood_budget_per_graph=0.1  # 🔧 可调整OOD预算
    )
    
    return dataset_train_id, (dataset_test_id, dataset_ood)
```

### 步骤2：在 `load_dataset` 函数中注册您的数据集

```python
def load_dataset(args, sub_dataname=''):
    # ... existing code ...
    elif args.dataset == 'your_dataset_name':  # 🔧 替换为您的数据集名称
        dataset_ind, dataset_ood_te = load_your_dgl_dataset(args)
    # ... existing code ...
```

## 2. 数据集要求和格式

### DGL图数据要求：
- **节点特征**: `graph.ndata['feature']` - 形状: `[num_nodes, feature_dim]`
- **节点标签**: `graph.ndata['label']` - 形状: `[num_nodes]`，标签从0开始编码
- **边信息**: `graph.edges()` - 返回源节点和目标节点的索引

### 数据类型要求：
- 节点特征: `torch.float32`
- 节点标签: `torch.long`
- 标签编码: 从0开始，连续编码 (0, 1, 2, ..., num_classes-1)

## 3. 运行步骤

### 步骤1：训练基础GNN模型
```bash
python train_id.py --dataset your_dataset_name --method gcn --device 0 --runs 5
```

### 步骤2：运行GRASP OOD检测
```bash
python test_ood.py --dataset your_dataset_name --method gcn --ood GRASP --device 0 --runs 5
```

### 步骤3：运行其他baseline方法
```bash
# MSP方法
python test_ood.py --dataset your_dataset_name --method gcn --ood MSP --device 0 --runs 5

# Energy方法  
python test_ood.py --dataset your_dataset_name --method gcn --ood Energy --device 0 --runs 5

# 其他方法...
```

## 4. 参数配置

### 在 `hyparams.py` 中添加您的数据集配置（可选）：

```python
hparams = {
    # ... existing configs ...
    'your_dataset_name': {
        'gcn': {
            'num_layers': 2, 
            'hidden_channels': 64, 
            'lr': 0.01,
            'weight_decay': 0.0005,
            'dropout': 0.5
        },
        'gat': {
            'num_layers': 2,
            'hidden_channels': 32,
            'gat_heads': 8,
            'lr': 0.01,
            'dropout': 0.5
        }
        # 添加其他方法的配置...
    }
}
```

## 5. 常见问题和解决方案

### 问题1：标签不从0开始
```python
# 在load_your_dgl_dataset函数中添加标签重映射
unique_labels = torch.unique(node_labels)
label_map = {label.item(): i for i, label in enumerate(unique_labels)}
node_labels = torch.tensor([label_map[label.item()] for label in node_labels])
```

### 问题2：图不连通
```python
# 检查图的连通性
from torch_geometric.utils import is_undirected, to_undirected

if not is_undirected(edge_index):
    edge_index = to_undirected(edge_index)
```

### 问题3：内存不足
```python
# 在args中设置较小的batch size或使用采样
args.batch_size = 1000  # 根据您的GPU内存调整
```

## 6. 验证数据集加载

创建一个简单的测试脚本验证数据集是否正确加载：

```python
import torch
from dataset import load_dataset
import argparse

class Args:
    def __init__(self):
        self.dataset = 'your_dataset_name'
        self.ood_type = 'structure'
        self.seed = 0
        self.train_prop = 0.1
        self.valid_prop = 0.1

args = Args()
dataset_ind, dataset_ood_te = load_dataset(args)

print(f"Dataset loaded successfully!")
print(f"ID dataset - Nodes: {dataset_ind.num_nodes}, Features: {dataset_ind.x.shape[1]}, Classes: {dataset_ind.y.max().item() + 1}")

if isinstance(dataset_ood_te, tuple):
    dataset_test_id, dataset_ood = dataset_ood_te
    print(f"Structure OOD detected")
    print(f"ID test nodes: {len(dataset_test_id.node_idx)}")
    print(f"OOD nodes: {len(dataset_ood.node_idx)}")
else:
    print(f"OOD nodes: {len(dataset_ood_te.node_idx)}")
```

## 7. 预期输出

成功运行后，您应该看到类似的输出：
```
Structure OOD detected: ID test nodes 1234, OOD nodes 567
Dataset: your_dataset_name  AUROC: 85.32  AUPR: 78.45  FPR95: 12.34
=======GRASP, time = 2.34567==========
```

结果将保存在 `results/your_dataset_name-gcn.csv` 文件中。