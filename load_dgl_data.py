import torch
import numpy as np
from dgl import load_graphs
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def load_dgl_dataset(data_path, dataset_name):
    """
    加载DGL格式的数据集并转换为torch_geometric.data.Data格式
    
    Args:
        data_path: 数据文件的路径前缀
        dataset_name: 数据集名称
    
    Returns:
        dataset_ind: 用于ID检测的数据
        dataset_ood_te: 用于OOD检测的数据 (这里返回None，因为你已经有预训练结果)
    """
    # 加载DGL图数据
    graph = load_graphs(data_path + dataset_name)[0][0]
    
    # 提取节点特征、标签和边
    node_feats = graph.ndata['feature']
    node_labels = graph.ndata['label']
    edge_index = graph.edges()
    
    # 转换边索引格式 (DGL格式转为torch_geometric格式)
    edge_index = torch.stack([edge_index[0], edge_index[1]], dim=0)
    
    # 确保边是无向的
    edge_index = to_undirected(edge_index)
    
    # 创建torch_geometric Data对象
    dataset_ind = Data(
        x=node_feats,
        edge_index=edge_index,
        y=node_labels
    )
    
    # 添加节点索引
    dataset_ind.node_idx = torch.arange(node_feats.shape[0])
    dataset_ind.num_nodes = node_feats.shape[0]
    dataset_ind.num_node_features = node_feats.shape[1]
    
    # 对于你的情况，可能不需要OOD测试数据，因为你已经有预训练的logits
    # 这里返回None，或者你可以根据需要创建OOD数据
    dataset_ood_te = None
    
    return dataset_ind, dataset_ood_te


def create_custom_ood_split(dataset_ind, ood_ratio=0.2, ood_type='random'):
    """
    创建自定义的OOD分割
    
    Args:
        dataset_ind: 原始数据集
        ood_ratio: OOD数据的比例
        ood_type: OOD分割类型 ('random', 'label_based', 等)
    
    Returns:
        dataset_ind: 更新后的ID数据
        dataset_ood_te: OOD测试数据
    """
    num_nodes = dataset_ind.num_nodes
    num_ood = int(num_nodes * ood_ratio)
    
    if ood_type == 'random':
        # 随机选择OOD节点
        perm = torch.randperm(num_nodes)
        ood_idx = perm[:num_ood]
        id_idx = perm[num_ood:]
    elif ood_type == 'label_based':
        # 基于标签选择OOD节点 (例如选择某些类别作为OOD)
        # 这里假设最后一个类别作为OOD
        max_label = dataset_ind.y.max().item()
        ood_mask = dataset_ind.y == max_label
        ood_idx = torch.where(ood_mask)[0]
        id_idx = torch.where(~ood_mask)[0]
    else:
        raise ValueError(f"Unsupported ood_type: {ood_type}")
    
    # 创建ID数据集
    dataset_ind.node_idx = id_idx
    
    # 创建OOD数据集
    dataset_ood_te = Data(
        x=dataset_ind.x,
        edge_index=dataset_ind.edge_index,
        y=dataset_ind.y
    )
    dataset_ood_te.node_idx = ood_idx
    
    return dataset_ind, dataset_ood_te