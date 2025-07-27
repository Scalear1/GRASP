"""
数据集加载模块 - 只需要修改标记🔧的地方
"""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, subgraph
from data_utils import set_random_seed, rand_splits

def load_dataset(args, sub_dataname=''):
    """
    数据集加载函数 - 只需要在这里添加您的数据集
    """
    if args.dataset == 'your_dataset_name':  # 🔧 修改为您的数据集名称
        dataset_ind, dataset_ood_te = load_your_dgl_dataset(args)
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')
    
    return dataset_ind, dataset_ood_te

def load_your_dgl_dataset(args):
    """
    加载您的DGL数据集 - 需要修改标记🔧的地方
    """
    from dgl import load_graphs
    
    # 🔧 修改1: 设置您的数据集路径和文件名
    dataset_path = "/path/to/your/dataset/your_file.bin"  # 完整的文件路径
    
    # 🔧 修改2: 如果您的特征/标签字段名不同，请修改
    feature_key = 'feature'  # 节点特征的字段名
    label_key = 'label'      # 节点标签的字段名
    
    # 加载DGL图数据
    graph = load_graphs(dataset_path)[0][0] 
    node_feats = graph.ndata[feature_key]
    node_labels = graph.ndata[label_key]
    edge_index = graph.edges()
    
    # 转换为PyTorch Geometric格式
    edge_index = torch.stack([edge_index[0], edge_index[1]], dim=0)
    
    # 数据类型转换
    if node_feats.dtype != torch.float32:
        node_feats = node_feats.float()
    if node_labels.dtype != torch.long:
        node_labels = node_labels.long()
    
    # 🔧 修改3: 如果您的标签不是从0开始，取消下面的注释
    # # 重新映射标签从0开始
    # unique_labels = torch.unique(node_labels)
    # label_map = {label.item(): i for i, label in enumerate(unique_labels)}
    # node_labels = torch.tensor([label_map[label.item()] for label in node_labels])
    
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
    dataset_train_id, dataset_test_id, dataset_ood = create_structure_ood(
        data, run, args, ood_ratio=0.1  # 10%的节点作为OOD
    )
    
    return dataset_train_id, (dataset_test_id, dataset_ood)

def create_structure_ood(data, run, args, ood_ratio=0.1):
    """
    创建结构OOD数据 - 无需修改
    """
    num_nodes = data.num_nodes
    num_ood = int(num_nodes * ood_ratio)
    
    # 随机选择OOD节点
    set_random_seed(run + args.seed)
    all_indices = torch.arange(num_nodes)
    ood_indices = torch.randperm(num_nodes)[:num_ood]
    id_indices = torch.tensor([i for i in range(num_nodes) if i not in ood_indices.tolist()])
    
    # 分割ID节点为训练和测试
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