#!/usr/bin/env python3
"""
测试脚本：在DGL格式数据集上使用GRASP进行OOD检测
使用预训练好的GNN模型的logits结果

使用方法:
python test_dgl_dataset.py --data_path /path/to/your/data/ --dataset_name your_dataset.bin --logits_path /path/to/logits.pt --create_ood_split --ood_ratio 0.2
"""

import argparse
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, to_scipy_sparse_matrix, coalesce

from logger import Logger
from dataset import load_dataset
from baselines import *
from grasp import GRASP
from data_utils import rand_splits, eval_acc, eval_rocauc, set_random_seed, evaluate_ood
from parse import parse_method, parser_add_main_args
from hyparams import hparams
import faulthandler; faulthandler.enable()


def main():
    # 解析参数
    parser = argparse.ArgumentParser(description='GRASP Testing on DGL Dataset')
    parser_add_main_args(parser)
    
    # 添加GRASP特定参数
    parser.add_argument('--ood', type=str, default='GRASP', help='OOD detection method')
    parser.add_argument('--T', type=float, default=1.0, help='temperature for energy score')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for GRASP')
    parser.add_argument('--K', type=int, default=5, help='number of propagation steps')
    parser.add_argument('--delta', type=float, default=1.0, help='delta for GRASP')
    parser.add_argument('--tau1', type=float, default=20.0, help='tau1 for GRASP')
    parser.add_argument('--tau2', type=float, default=50.0, help='tau2 for GRASP')
    parser.add_argument('--st', type=str, default='top', choices=['top', 'low', 'random', 'test'], help='selection strategy')
    parser.add_argument('--col', action='store_true', help='use col to count connections')
    parser.add_argument('--adj1', action='store_true', help='use adj1')
    parser.add_argument('--test', action='store_true', help='test mode')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    
    print(f"Using device: {device}")
    print(f"Arguments: {args}")
    
    # 验证必要参数
    if not args.data_path or not args.dataset_name:
        raise ValueError("必须提供 --data_path 和 --dataset_name 参数")
    
    if not args.logits_path:
        raise ValueError("必须提供 --logits_path 参数，指向预训练模型的logits文件")
    
    # 设置数据集名称为custom-dgl
    args.dataset = 'custom-dgl'
    
    print("正在加载DGL数据集...")
    
    ### 加载和预处理数据 ###
    try:
        dataset_ind, dataset_ood_te = load_dataset(args)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    if dataset_ood_te is None:
        print("警告: 没有OOD测试数据，GRASP可能无法正常工作")
        print("建议使用 --create_ood_split 参数创建OOD分割")
        return
    
    edge_index = dataset_ind.edge_index
    num_nodes = dataset_ind.num_nodes
    ood_idx = dataset_ood_te.node_idx
    c = 2  # 根据你的描述，num_labels = 2
    d = dataset_ind.num_node_features
    
    print(f"数据集统计:")
    print(f"  节点数: {num_nodes}")
    print(f"  类别数: {c}")
    print(f"  节点特征维度: {d}")
    print(f"  边数: {edge_index.size(1)}")
    print(f"  ID节点数: {len(dataset_ind.node_idx)}")
    print(f"  OOD节点数: {len(ood_idx)}")
    
    ### 加载预训练的logits ###
    print(f"正在加载预训练logits: {args.logits_path}")
    try:
        logits = torch.load(args.logits_path, map_location=device)
        print(f"Logits形状: {logits.shape}")
        
        # 验证logits形状
        if logits.shape[0] != num_nodes:
            raise ValueError(f"Logits节点数 ({logits.shape[0]}) 与图节点数 ({num_nodes}) 不匹配")
        if logits.shape[1] != c:
            print(f"警告: Logits类别数 ({logits.shape[1]}) 与预期类别数 ({c}) 不匹配")
            c = logits.shape[1]  # 更新类别数
            
    except Exception as e:
        print(f"加载logits失败: {e}")
        return
    
    # 使用二分类的评估函数
    if c == 2:
        criterion = nn.BCEWithLogitsLoss()
        eval_func = eval_rocauc
    else:
        criterion = nn.NLLLoss()
        eval_func = eval_acc
    
    # 初始化GRASP
    grasp = GRASP(args)
    
    ### 测试 ###
    logger = Logger(args.runs, args)
    durations = []
    
    for run in range(args.runs):
        t = time.time()
        print(f'----开始第 {run+1} 次运行----')
        set_random_seed(run + args.seed)
        
        # 创建数据分割
        split_idx = rand_splits(dataset_ind.node_idx, train_prop=args.train_prop, valid_prop=args.valid_prop)
        
        print(f"数据分割:")
        print(f"  训练集: {len(split_idx['train'])}")
        print(f"  验证集: {len(split_idx['valid'])}")
        print(f"  测试集: {len(split_idx['test'])}")
        
        # 使用GRASP进行OOD检测
        print("正在运行GRASP...")
        try:
            train_val_idx = torch.concat([split_idx['train'], split_idx['valid']])
            scores = grasp.detect(
                logits, 
                dataset_ind, 
                train_val_idx, 
                split_idx['test'], 
                ood_idx, 
                args
            )
            
            scores = scores.to(device)
            iid_score = scores[split_idx['test']]
            ood_score = scores[ood_idx]
            
            # 评估结果
            result = evaluate_ood(iid_score, ood_score)[:-1]
            print(f'运行 {run+1} 结果: AUROC={result[0]:.4f}, AUPR={result[1]:.4f}, FPR95={result[2]:.4f}')
            logger.add_result(run, result)
            
        except Exception as e:
            print(f"GRASP运行失败: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        durations.append(time.time() - t)
    
    print(f'======GRASP平均运行时间: {np.array(durations).mean():.5f}秒==========')
    
    ### 保存结果 ###
    result = logger.print_statistics()
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    filename = f'results/custom-dgl-GRASP.csv'
    print(f"保存结果到: {filename}")
    
    with open(filename, 'a+') as write_obj:
        write_obj.write(f"custom-dgl,GRASP,GRASP," + 
                        f"{result[:, 0].mean():.2f} ± {result[:, 0].std():.2f}," +
                        f"{result[:, 1].mean():.2f} ± {result[:, 1].std():.2f}," +
                        f"{result[:, 2].mean():.2f} ± {result[:, 2].std():.2f}\n")
    
    print("测试完成！")


if __name__ == '__main__':
    main()