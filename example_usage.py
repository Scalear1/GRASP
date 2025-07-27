#!/usr/bin/env python3
"""
使用示例：展示如何在DGL数据集上使用GRASP

这个脚本展示了：
1. 如何准备你的数据格式
2. 如何保存预训练模型的logits
3. 如何运行GRASP测试

假设你已经有：
- DGL格式的图数据
- 预训练GNN模型产生的logits
"""

import torch
import numpy as np
from dgl import load_graphs

def prepare_logits_example():
    """
    示例：如何准备和保存预训练模型的logits
    
    假设你已经有预训练的GNN模型和DGL图数据
    """
    
    # 示例：假设你的数据加载方式
    # graph = load_graphs(prefix + name)[0][0]
    # node_feats = graph.ndata['feature']
    # node_labels = graph.ndata['label'] 
    # edge_index = graph.edges()
    # num_labels = 2
    
    # 假设你已经通过预训练模型得到了logits
    # 这里创建一个示例logits (实际使用时替换为你的真实logits)
    num_nodes = 1000  # 替换为你的实际节点数
    num_classes = 2   # 你提到num_labels = 2
    
    # 示例logits (替换为你的真实logits)
    logits = torch.randn(num_nodes, num_classes)
    
    # 保存logits以供GRASP使用
    torch.save(logits, 'your_logits.pt')
    print(f"已保存logits到 your_logits.pt，形状: {logits.shape}")
    
    return logits

def run_grasp_example():
    """
    示例：如何运行GRASP测试
    """
    import subprocess
    
    # 准备示例logits
    prepare_logits_example()
    
    # 构建运行命令
    cmd = [
        'python', 'test_dgl_dataset.py',
        '--data_path', '/path/to/your/data/',  # 替换为你的数据路径
        '--dataset_name', 'your_dataset.bin',  # 替换为你的数据集文件名
        '--logits_path', 'your_logits.pt',     # logits文件路径
        '--create_ood_split',                  # 创建OOD分割
        '--ood_ratio', '0.2',                  # OOD数据比例
        '--ood_type', 'random',                # OOD分割类型
        '--runs', '5',                         # 运行次数
        '--alpha', '0.1',                      # GRASP alpha参数
        '--K', '5',                            # 传播步数
        '--delta', '1.0',                      # GRASP delta参数
        '--tau1', '20.0',                      # GRASP tau1参数
        '--tau2', '50.0',                      # GRASP tau2参数
        '--st', 'top',                         # 选择策略
    ]
    
    print("运行命令:")
    print(' '.join(cmd))
    print("\n注意：请将上述命令中的路径替换为你的实际路径")
    
    # 如果你想直接运行（取消注释下面的行）
    # subprocess.run(cmd)

def main():
    print("=== DGL数据集上使用GRASP的完整示例 ===")
    print()
    
    print("步骤1: 准备你的数据")
    print("确保你有以下文件：")
    print("- DGL格式的图数据文件 (例如: your_dataset.bin)")
    print("- 预训练GNN模型产生的logits (例如: your_logits.pt)")
    print()
    
    print("步骤2: 数据格式要求")
    print("你的DGL图应该包含：")
    print("- graph.ndata['feature']: 节点特征")
    print("- graph.ndata['label']: 节点标签")
    print("- graph.edges(): 边连接")
    print()
    
    print("步骤3: 准备logits文件")
    print("logits应该是形状为 [num_nodes, num_classes] 的torch.Tensor")
    prepare_logits_example()
    print()
    
    print("步骤4: 运行GRASP测试")
    run_grasp_example()
    print()
    
    print("步骤5: 实际使用时的修改")
    print("请修改以下参数：")
    print("- --data_path: 你的数据文件路径前缀")
    print("- --dataset_name: 你的DGL数据集文件名")
    print("- --logits_path: 你的预训练logits文件路径")
    print()
    
    print("可选的GRASP参数调优：")
    print("- --alpha: 控制信息传播的权重 (默认0.1)")
    print("- --K: 传播步数 (默认5)")
    print("- --delta: 增强节点的权重 (默认1.0)")
    print("- --tau1, --tau2: 节点选择的阈值参数")
    print("- --st: 节点选择策略 ('top', 'low', 'random', 'test')")
    print()
    
    print("完整的运行示例：")
    print("python test_dgl_dataset.py \\")
    print("    --data_path /your/data/path/ \\")
    print("    --dataset_name your_dataset.bin \\")
    print("    --logits_path your_logits.pt \\")
    print("    --create_ood_split \\")
    print("    --ood_ratio 0.2 \\")
    print("    --runs 5 \\")
    print("    --alpha 0.1 \\")
    print("    --K 5")

if __name__ == '__main__':
    main()