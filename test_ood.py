import argparse
import sys
import os
import time
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, to_scipy_sparse_matrix, coalesce

from logger import Logger
from dataset import load_dataset
from baselines import *
from grasp import GRASP
#from correct_smooth import double_correlation_autoscale, double_correlation_fixed
from data_utils import rand_splits, eval_acc, eval_rocauc, set_random_seed, evaluate_ood
from parse import parse_method, parser_add_main_args
from hyparams import hparams
import faulthandler; faulthandler.enable()


# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
np.random.seed(0)

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
# args_dict = vars(args)
if args.dataset in hparams:
    for hname, v in hparams[args.dataset][args.method].items():
        setattr(args, hname, v)
print(args)

if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')

### Load and preprocess data ###
dataset_ind, dataset_ood_te = load_dataset(args)

# 处理不同类型的OOD数据集格式
if isinstance(dataset_ood_te, tuple):
    # Structure OOD: dataset_ood_te = (dataset_test_id, dataset_ood)
    dataset_test_id, dataset_ood = dataset_ood_te
    ood_idx = dataset_ood.node_idx
    print(f"Structure OOD detected: ID test nodes {len(dataset_test_id.node_idx)}, OOD nodes {len(dataset_ood.node_idx)}")
else:
    # Feature OOD or Label OOD: dataset_ood_te is a single dataset
    ood_idx = dataset_ood_te.node_idx
    print(f"Feature/Label OOD detected: OOD nodes {len(ood_idx)}")

edge_index = dataset_ind.edge_index
num_nodes = dataset_ind.num_nodes
c = dataset_ind.y.max().item() + 1
d = dataset_ind.num_node_features

print(f"num nodes {num_nodes} | num classes {c} | num node feats {d}")

# using rocauc as the eval function
if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
    criterion = nn.BCEWithLogitsLoss()
    eval_func = eval_rocauc
else:
    criterion = nn.NLLLoss()
    eval_func = eval_acc

logger = Logger(args.runs, args)

# 设置logits文件路径
if args.logits_file:
    # 使用指定的单个logits文件（适用于所有runs）
    logits_path = args.logits_file
    print(f"Using single logits file: {logits_path}")
elif args.logits_dir:
    # 使用指定的logits目录
    logits_dir = args.logits_dir
    print(f"Looking for logits in: {logits_dir}")
else:
    # 使用默认路径
    model_path = f'{args.dataset}-{args.sub_dataset}' if args.sub_dataset else f'{args.dataset}'
    logits_dir = f'logits/{model_path}/{args.method}'
    print(f"Looking for logits in default path: {logits_dir}")

ood = eval(args.ood)(args)
### Testing ###
durations = []
for run in range(args.runs):
    t = time.time()
    print(f'----start time: {t}')
    set_random_seed(run + args.seed)
    split_idx = rand_splits(dataset_ind.node_idx, train_prop=args.train_prop, valid_prop=args.valid_prop)

    # 加载预计算的logits
    if args.logits_file:
        # 使用单个logits文件（所有runs共用）
        logit_path = args.logits_file
    else:
        # 使用按run编号的logits文件
        logit_path = f'{logits_dir}/logit{run}.pt'
    
    if os.path.exists(logit_path):
        print(f'Loading logits from: {logit_path}')
        logit = torch.load(logit_path, map_location=device)
        print(f'Logits shape: {logit.shape}')
    else:
        print(f"❌ Logits file not found: {logit_path}")
        if args.force_logits:
            print("Exiting because --force_logits is enabled.")
            sys.exit(1)
        else:
            print("Skipping this run...")
            continue
    
    if args.ood in ['MSP', 'Energy', 'ODIN']:
        scores = ood.detect(logit)
    elif args.ood == 'GNNSafe':
        scores = ood.detect(logit, dataset_ind.edge_index, args)
    elif args.ood == 'Mahalanobis':
        scores = ood.detect(logit, torch.concat([split_idx['train'], split_idx['valid']]), torch.concat([split_idx['test'], dataset_ood_te.node_idx]), dataset_ind.y)
    elif args.ood == 'KNN':
        score_ckpt = f'{logits_dir}/score{run}.pt'
        if os.path.exists(score_ckpt):
            scores = torch.load(score_ckpt, map_location='cpu')
        else:
            scores = ood.detect(logit, torch.concat([split_idx['train'], split_idx['valid']]))
            torch.save(scores, score_ckpt)
    elif args.ood == 'GRASP':
        scores = ood.detect(logit, dataset_ind, torch.concat([split_idx['train'], split_idx['valid']]), split_idx['test'], ood_idx, args)

    scores = scores.to(device)
    
    # 根据OOD类型选择正确的ID测试集
    if isinstance(dataset_ood_te, tuple):
        # Structure OOD: 使用预定义的ID测试集
        dataset_test_id, dataset_ood = dataset_ood_te
        # 需要将scores映射到正确的节点索引
        # 对于structure OOD，我们需要分别评估ID测试集和OOD测试集
        
        # 获取ID测试集的分数 - 这需要根据具体的节点映射来处理
        # 由于structure_shift_dataset改变了节点索引，我们需要特殊处理
        print("Warning: Structure OOD evaluation may need additional index mapping")
        iid_score = scores[split_idx['test']]  # 使用原始的测试分割
        ood_score = scores[ood_idx]
    else:
        # Feature/Label OOD: 使用标准的测试分割
        iid_score = scores[split_idx['test']]
        ood_score = scores[ood_idx]
        
    result = evaluate_ood(iid_score, ood_score)[:-1]
    print(f'{args.dataset}'+'\t'.join([str(x) for x in result]))
    logger.add_result(run, result)
    durations.append(time.time()-t)


ood_name = args.ood
print(f'======={ood_name}, time = {np.array(durations).mean():.5f}==========')

### Save results ###
result = logger.print_statistics()
filename = f'results/{args.dataset}-{args.method}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    write_obj.write(f"{args.dataset},"+ f"{args.method},{ood_name}," + 
                    f"{result[:, 0].mean():.2f} ± {result[:, 0].std():.2f}," +
                    f"{result[:, 1].mean():.2f} ± {result[:, 1].std():.2f}," +
                    f"{result[:, 2].mean():.2f} ± {result[:, 2].std():.2f}\n")
