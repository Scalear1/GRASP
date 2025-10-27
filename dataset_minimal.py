"""
简化的数据集加载模块 - 专门用于预计算logits的OOD检测
"""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from data_utils import set_random_seed

def load_dataset(args, sub_dataname=''):
    """
    简化的数据集加载函数
    只需要实现您使用的数据集
    """
    if args.dataset == 'your_dgl_dataset':  # 🔧 修改为您的数据集名称
        dataset_ind, dataset_ood_te = load_your_dgl_dataset(args)
    elif args.dataset == 'cora':  # 示例：如果您也想测试cora
        dataset_ind, dataset_ood_te = load_cora_example()
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')
    
    return dataset_ind, dataset_ood_te

def load_your_dgl_dataset(args):
    """
    加载您的DGL格式数据集
    🔧 请根据您的实际情况修改
    """
    from dgl import load_graphs
    
    # 🔧 修改为您的数据集路径和文件名
    prefix = "/path/to/your/dataset/"
    name = f"{args.dataset}.bin"
    
    # 加载DGL图数据
    graph = load_graphs(prefix + name)[0][0] 
    node_feats = graph.ndata['feature']  # 🔧 如果字段名不同请修改
    node_labels = graph.ndata['label']   # 🔧 如果字段名不同请修改
    edge_index = graph.edges()
    
    # 转换为PyTorch Geometric格式
    edge_index = torch.stack([edge_index[0], edge_index[1]], dim=0)
    
    # 数据类型转换
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
    data.node_idx = torch.arange(data.num_nodes)
    
    # 生成结构偏移的OOD数据
    run = getattr(args, 'run', 0)
    dataset_train_id, dataset_test_id, dataset_ood = structure_shift_dataset(
        data, run, args, ood_budget_per_graph=0.1
    )
    
    return dataset_train_id, (dataset_test_id, dataset_ood)

def load_cora_example():
    """
    简单的Cora数据集加载示例（用于测试）
    如果您不需要可以删除
    """
    from torch_geometric.datasets import Planetoid
    
    dataset = Planetoid(root='data/Planetoid', name='Cora')
    data = dataset[0]
    
    # 转换格式
    data.node_idx = torch.arange(data.num_nodes)
    
    # 生成简单的特征噪声OOD数据
    ood_data = create_feat_noise_dataset(data)
    
    return data, ood_data

def create_feat_noise_dataset(data):
    """
    创建特征噪声OOD数据集
    """
    x = data.x
    n = data.num_nodes
    
    torch.manual_seed(111)
    idx = torch.randint(0, n, (n, 2))
    torch.manual_seed(222)
    weight = torch.rand(n).unsqueeze(1)
    x_new = x[idx[:, 0]] * weight + x[idx[:, 1]] * (1 - weight)

    dataset = Data(x=x_new, edge_index=data.edge_index, y=data.y)
    dataset.node_idx = torch.arange(n)
    
    return dataset

def structure_shift_dataset(data, run, args, ood_budget_per_graph=0.1):
    """
    简化的结构偏移数据集生成
    这是一个基本实现，您可能需要根据需要调整
    """
    from torch_geometric.utils import subgraph, to_undirected
    from torch_geometric.utils.map import map_index
    from data_utils import rand_splits
    import numpy as np
    
    # 简化版本：随机选择一些节点作为OOD
    num_nodes = data.num_nodes
    num_ood = int(num_nodes * ood_budget_per_graph)
    
    set_random_seed(run + args.seed)
    all_indices = torch.arange(num_nodes)
    ood_indices = torch.randperm(num_nodes)[:num_ood]
    id_indices = torch.tensor([i for i in range(num_nodes) if i not in ood_indices.tolist()])
    
    # 创建ID训练数据集
    split_idx = rand_splits(id_indices, args.train_prop, args.valid_prop)
    train_idx = torch.concat([split_idx['train'], split_idx['valid']])
    test_id_idx = split_idx['test']
    
    # 创建子图
    train_edge_index = subgraph(train_idx, data.edge_index, relabel_nodes=True)[0]
    test_id_edge_index = subgraph(test_id_idx, data.edge_index, relabel_nodes=True)[0]
    ood_edge_index = subgraph(ood_indices, data.edge_index, relabel_nodes=True)[0]
    
    # 创建数据集
    dataset_train_id = Data(
        x=data.x[train_idx], 
        edge_index=train_edge_index, 
        y=data.y[train_idx]
    )
    dataset_train_id.node_idx = torch.arange(len(train_idx))
    
    dataset_test_id = Data(
        x=data.x[test_id_idx], 
        edge_index=test_id_edge_index, 
        y=data.y[test_id_idx]
    )
    dataset_test_id.node_idx = torch.arange(len(test_id_idx))
    
    dataset_ood = Data(
        x=data.x[ood_indices], 
        edge_index=ood_edge_index, 
        y=data.y[ood_indices]
    )
    dataset_ood.node_idx = torch.arange(len(ood_indices))
    
    return dataset_train_id, dataset_test_id, dataset_ood