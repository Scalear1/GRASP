#!/usr/bin/env python3
"""
最小化的OOD检测脚本 - 专门用于预计算logits
只包含必需的依赖文件
"""

import argparse
import sys
import os
import time
import numpy as np
import torch

from logger import Logger
from dataset_minimal import load_dataset  # 使用简化的数据集加载
from baselines import *
from grasp import GRASP
from data_utils import rand_splits, eval_acc, eval_rocauc, set_random_seed, evaluate_ood
from parse_minimal import parser_add_main_args  # 使用简化的参数解析

# 设置随机种子
np.random.seed(0)

### Parse args ###
parser = argparse.ArgumentParser(description='Minimal OOD Detection with Precomputed Logits')
parser_add_main_args(parser)
args = parser.parse_args()

print("=== Minimal OOD Detection with Precomputed Logits ===")
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
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("💡 请确保在 dataset_minimal.py 中正确配置了您的数据集加载函数")
    sys.exit(1)

# 处理不同类型的OOD数据集格式
if isinstance(dataset_ood_te, tuple):
    dataset_test_id, dataset_ood = dataset_ood_te
    ood_idx = dataset_ood.node_idx
    print(f"Structure OOD detected: ID test nodes {len(dataset_test_id.node_idx)}, OOD nodes {len(dataset_ood.node_idx)}")
else:
    ood_idx = dataset_ood_te.node_idx
    print(f"Feature/Label OOD detected: OOD nodes {len(ood_idx)}")

# 数据集基本信息
num_nodes = dataset_ind.num_nodes
c = dataset_ind.y.max().item() + 1
d = dataset_ind.num_node_features

print(f"Dataset info: {num_nodes} nodes | {c} classes | {d} features")

# 设置评估函数
if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

# 设置logits文件路径
if args.logits_file:
    print(f"Using single logits file: {args.logits_file}")
elif args.logits_dir:
    print(f"Looking for logits in: {args.logits_dir}")
else:
    # 使用默认路径
    model_path = f'{args.dataset}-{args.sub_dataset}' if args.sub_dataset else f'{args.dataset}'
    logits_dir = f'logits/{model_path}/{args.method}'
    print(f"Looking for logits in default path: {logits_dir}")

# 创建OOD检测器
try:
    ood = eval(args.ood)(args)
except Exception as e:
    print(f"❌ Error creating OOD detector '{args.ood}': {e}")
    print("💡 支持的OOD方法: MSP, Energy, ODIN, Mahalanobis, KNN, GNNSafe, GRASP")
    sys.exit(1)

### Testing ###
durations = []
successful_runs = 0

for run in range(args.runs):
    t = time.time()
    print(f'\n--- Run {run+1}/{args.runs} ---')
    
    set_random_seed(run + args.seed)
    split_idx = rand_splits(dataset_ind.node_idx, train_prop=args.train_prop, valid_prop=args.valid_prop)

    # 加载预计算的logits
    if args.logits_file:
        logit_path = args.logits_file
    else:
        logit_path = f'{logits_dir}/logit{run}.pt'
    
    if os.path.exists(logit_path):
        print(f'✓ Loading logits from: {logit_path}')
        try:
            logit = torch.load(logit_path, map_location=device)
            print(f'  Logits shape: {logit.shape}')
            
            # 验证logits维度
            expected_shape = (num_nodes, c)
            if logit.shape != expected_shape:
                print(f'❌ Logits shape mismatch! Expected {expected_shape}, got {logit.shape}')
                if args.force_logits:
                    sys.exit(1)
                else:
                    continue
        except Exception as e:
            print(f'❌ Error loading logits: {e}')
            if args.force_logits:
                sys.exit(1)
            else:
                continue
                
    else:
        print(f"❌ Logits file not found: {logit_path}")
        if args.force_logits:
            print("Exiting because --force_logits is enabled.")
            sys.exit(1)
        else:
            print("Skipping this run...")
            continue
    
    # 应用OOD检测方法
    print(f'  Applying {args.ood} method...')
    
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
            print(f"❌ Unknown OOD method: {args.ood}")
            continue
    except Exception as e:
        print(f"❌ Error in OOD detection: {e}")
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
        print(f'  Results - AUROC: {result[0]:.4f}, AUPR: {result[1]:.4f}, FPR95: {result[2]:.4f}')
        
        logger.add_result(run, result)
        durations.append(time.time()-t)
        successful_runs += 1
    except Exception as e:
        print(f"❌ Error in evaluation: {e}")
        continue

if successful_runs == 0:
    print("❌ No successful runs! Please check your setup.")
    sys.exit(1)

# 打印最终结果
ood_name = args.ood
print(f'\n=== Final Results ===')
print(f'Method: {ood_name}')
print(f'Successful runs: {successful_runs}/{args.runs}')
print(f'Average time per run: {np.array(durations).mean():.5f}s')

### Save results ###
result = logger.print_statistics()
filename = f'results/{args.dataset}-{args.method}.csv'
print(f"Saving results to {filename}")

# 确保results目录存在
os.makedirs('results', exist_ok=True)

with open(f"{filename}", 'a+') as write_obj:
    sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    write_obj.write(f"{args.dataset},"+ f"{args.method},{ood_name}," + 
                    f"{result[:, 0].mean():.2f} ± {result[:, 0].std():.2f}," +
                    f"{result[:, 1].mean():.2f} ± {result[:, 1].std():.2f}," +
                    f"{result[:, 2].mean():.2f} ± {result[:, 2].std():.2f}\n")

print("✅ Completed successfully!")