#!/usr/bin/env python3
"""
最终版本 - 使用预计算logits的OOD检测脚本
"""

import argparse
import sys
import os
import time
import numpy as np
import torch

from logger import Logger
from dataset_minimal_final import load_dataset  # 使用我们修改的数据集加载
from baselines import *
from grasp import GRASP
from data_utils import rand_splits, eval_acc, eval_rocauc, set_random_seed, evaluate_ood
from parse_minimal import parser_add_main_args

# 设置随机种子
np.random.seed(0)

### Parse args ###
parser = argparse.ArgumentParser(description='OOD Detection with Precomputed Logits')
parser_add_main_args(parser)
args = parser.parse_args()

print("=== GRASP OOD Detection with Precomputed Logits ===")
print(f"Dataset: {args.dataset}")
print(f"OOD Method: {args.ood}")
print(f"Runs: {args.runs}")

if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')

print(f"Device: {device}")

### Load and preprocess data ###
try:
    dataset_ind, dataset_ood_te = load_dataset(args)
    print("✅ 数据集加载成功")
except Exception as e:
    print(f"❌ 数据集加载失败: {e}")
    print("💡 请检查 dataset_minimal_final.py 中的路径和字段名设置")
    sys.exit(1)

# 处理不同类型的OOD数据集格式
if isinstance(dataset_ood_te, tuple):
    dataset_test_id, dataset_ood = dataset_ood_te
    ood_idx = dataset_ood.node_idx
    print(f"Structure OOD: ID测试节点 {len(dataset_test_id.node_idx)}, OOD节点 {len(dataset_ood.node_idx)}")
else:
    ood_idx = dataset_ood_te.node_idx
    print(f"Feature/Label OOD: OOD节点 {len(ood_idx)}")

# 数据集基本信息
num_nodes = dataset_ind.num_nodes
c = dataset_ind.y.max().item() + 1
d = dataset_ind.num_node_features

print(f"数据集信息: {num_nodes} 个节点 | {c} 个类别 | {d} 维特征")

# 设置评估函数
if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

# 设置logits文件路径
if args.logits_file:
    print(f"使用单个logits文件: {args.logits_file}")
elif args.logits_dir:
    print(f"在目录中查找logits: {args.logits_dir}")
else:
    # 使用默认路径
    model_path = f'{args.dataset}-{args.sub_dataset}' if args.sub_dataset else f'{args.dataset}'
    logits_dir = f'logits/{model_path}/{args.method}'
    print(f"在默认路径查找logits: {logits_dir}")

# 创建OOD检测器
try:
    ood = eval(args.ood)(args)
    print(f"✅ 创建 {args.ood} 检测器成功")
except Exception as e:
    print(f"❌ 创建OOD检测器失败: {e}")
    print("💡 支持的方法: MSP, Energy, ODIN, Mahalanobis, KNN, GNNSafe, GRASP")
    sys.exit(1)

### Testing ###
durations = []
successful_runs = 0

for run in range(args.runs):
    t = time.time()
    print(f'\n--- 第 {run+1}/{args.runs} 次运行 ---')
    
    set_random_seed(run + args.seed)
    split_idx = rand_splits(dataset_ind.node_idx, train_prop=args.train_prop, valid_prop=args.valid_prop)

    # 加载预计算的logits
    if args.logits_file:
        logit_path = args.logits_file
    else:
        logit_path = f'{logits_dir}/logit{run}.pt'
    
    if os.path.exists(logit_path):
        print(f'✅ 从以下路径加载logits: {logit_path}')
        try:
            logit = torch.load(logit_path, map_location=device)
            print(f'   Logits形状: {logit.shape}')
            
            # 验证logits维度
            expected_shape = (num_nodes, c)
            if logit.shape != expected_shape:
                print(f'❌ Logits形状不匹配! 期望 {expected_shape}, 实际 {logit.shape}')
                if args.force_logits:
                    sys.exit(1)
                else:
                    continue
        except Exception as e:
            print(f'❌ 加载logits失败: {e}')
            if args.force_logits:
                sys.exit(1)
            else:
                continue
                
    else:
        print(f"❌ 找不到logits文件: {logit_path}")
        if args.force_logits:
            print("因为启用了 --force_logits，程序退出")
            sys.exit(1)
        else:
            print("跳过此次运行...")
            continue
    
    # 应用OOD检测方法
    print(f'   应用 {args.ood} 方法...')
    
    try:
        if args.ood in ['MSP', 'Energy', 'ODIN']:
            scores = ood.detect(logit)
        elif args.ood == 'GNNSafe':
            scores = ood.detect(logit, dataset_ind.edge_index, args)
        elif args.ood == 'Mahalanobis':
            scores = ood.detect(logit, torch.concat([split_idx['train'], split_idx['valid']]), 
                               torch.concat([split_idx['test'], dataset_ood_te.node_idx]), dataset_ind.y)
        elif args.ood == 'KNN':
            scores = ood.detect(logit, torch.concat([split_idx['train'], split_idx['valid']]))
        elif args.ood == 'GRASP':
            scores = ood.detect(logit, dataset_ind, torch.concat([split_idx['train'], split_idx['valid']]), 
                               split_idx['test'], ood_idx, args)
        else:
            print(f"❌ 未知的OOD方法: {args.ood}")
            continue
    except Exception as e:
        print(f"❌ OOD检测过程出错: {e}")
        if args.force_logits:
            sys.exit(1)
        else:
            continue

    scores = scores.to(device)
    
    # 根据OOD类型选择正确的ID测试集
    if isinstance(dataset_ood_te, tuple):
        dataset_test_id, dataset_ood = dataset_ood_te
        iid_score = scores[split_idx['test']]
        ood_score = scores[ood_idx]
    else:
        iid_score = scores[split_idx['test']]
        ood_score = scores[ood_idx]
        
    # 评估OOD检测性能
    try:
        result = evaluate_ood(iid_score, ood_score)[:-1]
        print(f'   结果 - AUROC: {result[0]:.4f}, AUPR: {result[1]:.4f}, FPR95: {result[2]:.4f}')
        
        logger.add_result(run, result)
        durations.append(time.time()-t)
        successful_runs += 1
    except Exception as e:
        print(f"❌ 评估过程出错: {e}")
        continue

if successful_runs == 0:
    print("❌ 没有成功的运行! 请检查设置")
    sys.exit(1)

# 打印最终结果
ood_name = args.ood
print(f'\n=== 最终结果 ===')
print(f'方法: {ood_name}')
print(f'成功运行: {successful_runs}/{args.runs}')
print(f'平均运行时间: {np.array(durations).mean():.5f}秒')

### Save results ###
result = logger.print_statistics()
filename = f'results/{args.dataset}-{args.method}.csv'
print(f"保存结果到: {filename}")

# 确保results目录存在
os.makedirs('results', exist_ok=True)

with open(f"{filename}", 'a+') as write_obj:
    sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    write_obj.write(f"{args.dataset},"+ f"{args.method},{ood_name}," + 
                    f"{result[:, 0].mean():.2f} ± {result[:, 0].std():.2f}," +
                    f"{result[:, 1].mean():.2f} ± {result[:, 1].std():.2f}," +
                    f"{result[:, 2].mean():.2f} ± {result[:, 2].std():.2f}\n")

print("🎉 完成!")