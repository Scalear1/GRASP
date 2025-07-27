import argparse
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, to_scipy_sparse_matrix, coalesce
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, subgraph, mask_to_index, map_index
from torch_sparse import SparseTensor, matmul
import dgl
from dgl.data import TFinancDataset

from logger import Logger
from baselines import *
from grasp import GRASP
from data_utils import rand_splits, eval_acc, eval_rocauc, set_random_seed, evaluate_ood
import faulthandler; faulthandler.enable()

def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def structure_shift_dataset(data, run, args, ood_budget_per_graph=0.1):
    """Create structure shift OOD dataset for tfinance"""
    seed = run + args.seed
    idx = torch.arange(data.num_nodes)

    # extract directed edges of undirected edge_index
    start, end = data.edge_index
    mask = start < end
    start, end = start[mask], end[mask]

    # extract nodes belonging to community
    num_classes = data.y.max() + 1
    nodes_per_class = [None] * num_classes
    for c in range(num_classes):
        nodes_per_class[c] = torch.where(data.y == c)[0].tolist()

    ood_idx = []
    ood_edge_index = []

    # for each community
    for c in range(num_classes):
        deleted_edges = []
        intra_community_edges = torch.where((data.y[start] == c) & (data.y[end] == c))[0].tolist()

        # find all nodes from other communities
        cross_community_nodes = []
        for c_j in range(num_classes):
            if c_j == c:
                continue
            cross_community_nodes.extend(nodes_per_class[c_j])

        # calculate budget
        n_intra_edges = len(intra_community_edges)
        budget = int(ood_budget_per_graph * n_intra_edges)

        # sample deleted
        set_seed(seed)
        deleted_edges.extend(np.random.choice(intra_community_edges, budget, replace=False))

        # first: sample community nodes, then sample corresponding cross-community nodes
        community_nodes = start[deleted_edges]
        ood_idx.extend(community_nodes.tolist())
        set_seed(seed)
        cross_community_nodes = np.random.choice(cross_community_nodes, budget, replace=True)
        ood_edge_index.append(torch.vstack([community_nodes, cross_community_nodes]))
        ood_idx.extend(cross_community_nodes.tolist())

    ood_edge_index = torch.concat(ood_edge_index, dim=1)
    ood_edge_index = to_undirected(ood_edge_index)
    ood_idx = np.unique(ood_idx)
    ood_idx = torch.as_tensor(ood_idx)
    ood_edge_index, _ = map_index(ood_edge_index.view(-1), ood_idx, inclusive=True)
    ood_edge_index = ood_edge_index.view(2, -1)
    dataset_ood = Data(x=data.x[ood_idx], edge_index=ood_edge_index, y=data.y[ood_idx])

    edge_index = data.edge_index
    id_mask = torch.ones(data.num_nodes, dtype=bool)
    id_mask[ood_idx] = 0
    id_train_idx = mask_to_index(id_mask)
    set_seed(seed)
    split_idx = rand_splits(id_train_idx, args.train_prop, args.valid_prop)
    test_id_idx = split_idx['test']
    test_id_edge_index = subgraph(test_id_idx, edge_index, relabel_nodes=True)[0]
    dataset_test_id = Data(x=data.x[test_id_idx], edge_index=test_id_edge_index, y=data.y[test_id_idx])

    train_idx = torch.concat([split_idx['train'], split_idx['valid']])
    train_id_edge_index = subgraph(train_idx, edge_index, relabel_nodes=True)[0]
    dataset_train_id = Data(x=data.x[train_idx], edge_index=train_id_edge_index, y=data.y[train_idx])
    new_split_idx = {'train': map_index(split_idx['train'], train_idx, inclusive=True)[0], 
                     'valid': map_index(split_idx['valid'], train_idx, inclusive=True)[0]}
    dataset_train_id.split_idx = new_split_idx

    return dataset_train_id, dataset_test_id, dataset_ood

def load_tfinance_data():
    """Load tfinance dataset and convert to PyTorch Geometric format"""
    # Load DGL graph
    graph = dgl.data.TFinancDataset()[0]
    
    # Extract features and labels
    node_feats = graph.ndata['feature']
    node_labels = graph.ndata['label']
    edge_index = graph.edges()
    
    # Convert to PyTorch Geometric format
    edge_index = torch.stack([edge_index[0], edge_index[1]], dim=0)
    
    # Create PyG Data object
    data = Data(x=node_feats, edge_index=edge_index, y=node_labels)
    
    return data

# Parse arguments
parser = argparse.ArgumentParser(description='OOD Detection on TFinance Dataset')
parser.add_argument('--device', type=int, default=0, help='GPU device')
parser.add_argument('--cpu', action='store_true', help='Use CPU')
parser.add_argument('--runs', type=int, default=5, help='Number of runs')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--ood', type=str, default='GRASP', help='OOD detection method')
parser.add_argument('--logits_path', type=str, required=True, help='Path to pre-trained logits')
parser.add_argument('--train_prop', type=float, default=0.5, help='Training proportion')
parser.add_argument('--valid_prop', type=float, default=0.25, help='Validation proportion')
parser.add_argument('--ood_budget', type=float, default=0.1, help='OOD budget for structure shift')

# GRASP specific parameters
parser.add_argument('--T', type=float, default=1.0, help='Temperature for Energy score')
parser.add_argument('--K', type=int, default=8, help='Number of propagation steps')
parser.add_argument('--alpha', type=float, default=0.1, help='Propagation weight')
parser.add_argument('--delta', type=float, default=0.1, help='Enhancement coefficient')
parser.add_argument('--tau1', type=float, default=10, help='Threshold for node selection')
parser.add_argument('--tau2', type=float, default=50, help='Percentage of nodes to select')
parser.add_argument('--st', type=str, default='top', help='Selection strategy')
parser.add_argument('--col', action='store_true', help='Use column-wise adjacency')
parser.add_argument('--adj1', action='store_true', help='Use original adjacency')
parser.add_argument('--test', action='store_true', help='Use test nodes for selection')

args = parser.parse_args()

# Set device
if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')

print(f"Using device: {device}")
print(f"Arguments: {args}")

# Load data
print("Loading TFinance dataset...")
data = load_tfinance_data()
print(f"Dataset loaded: {data.num_nodes} nodes, {data.edge_index.size(1)} edges, {data.num_node_features} features")

# Load pre-trained logits
print(f"Loading logits from {args.logits_path}")
logits = torch.load(args.logits_path, map_location=device)
print(f"Logits shape: {logits.shape}")

# Create structure shift OOD dataset
print("Creating structure shift OOD dataset...")
dataset_ind, dataset_test_id, dataset_ood_te = structure_shift_dataset(data, 0, args, args.ood_budget)

print(f"ID dataset: {dataset_ind.num_nodes} nodes, {dataset_ind.edge_index.size(1)} edges")
print(f"Test ID dataset: {dataset_test_id.num_nodes} nodes, {dataset_test_id.edge_index.size(1)} edges")
print(f"OOD dataset: {dataset_ood_te.num_nodes} nodes, {dataset_ood_te.edge_index.size(1)} edges")

# Initialize logger
logger = Logger(args.runs, args)

# Initialize OOD detector
ood_detector = eval(args.ood)(args)

# Testing
durations = []
for run in range(args.runs):
    t = time.time()
    print(f'----Run {run+1}/{args.runs}, start time: {t}')
    set_random_seed(run + args.seed)
    
    # Get logits for current run (assuming logits is a list or tensor)
    if isinstance(logits, list) and len(logits) > run:
        current_logits = logits[run]
    else:
        current_logits = logits
    
    # Ensure logits are on correct device
    current_logits = current_logits.to(device)
    
    # Detect OOD
    if args.ood in ['MSP', 'Energy', 'ODIN']:
        scores = ood_detector.detect(current_logits)
    elif args.ood == 'GNNSafe':
        scores = ood_detector.detect(current_logits, dataset_ind.edge_index, args)
    elif args.ood == 'Mahalanobis':
        train_idx = torch.concat([dataset_ind.split_idx['train'], dataset_ind.split_idx['valid']])
        test_idx = torch.concat([dataset_test_id.node_idx, dataset_ood_te.node_idx])
        scores = ood_detector.detect(current_logits, train_idx, test_idx, dataset_ind.y)
    elif args.ood == 'KNN':
        scores = ood_detector.detect(current_logits, dataset_ind.split_idx['train'])
    elif args.ood == 'GRASP':
        train_idx = torch.concat([dataset_ind.split_idx['train'], dataset_ind.split_idx['valid']])
        test_id_idx = dataset_test_id.node_idx
        test_ood_idx = dataset_ood_te.node_idx
        scores = ood_detector.detect(current_logits, dataset_ind, train_idx, test_id_idx, test_ood_idx, args)
    else:
        raise ValueError(f"Unknown OOD method: {args.ood}")
    
    # Evaluate OOD detection performance
    test_id_scores = scores[dataset_test_id.node_idx]
    test_ood_scores = scores[dataset_ood_te.node_idx]
    
    # Create labels: 0 for ID, 1 for OOD
    test_id_labels = torch.zeros(len(test_id_scores))
    test_ood_labels = torch.ones(len(test_ood_scores))
    
    # Concatenate scores and labels
    all_scores = torch.cat([test_id_scores, test_ood_scores])
    all_labels = torch.cat([test_id_labels, test_ood_labels])
    
    # Calculate metrics
    auroc, aupr = evaluate_ood(all_scores, all_labels)
    
    # Log results
    logger.add_result(run, (auroc, aupr))
    
    duration = time.time() - t
    durations.append(duration)
    print(f'Run {run+1} completed in {duration:.2f}s')
    print(f'AUROC: {auroc:.4f}, AUPR: {aupr:.4f}')

# Print final results
print('=' * 50)
print('Final Results:')
print(f'OOD Method: {args.ood}')
print(f'Dataset: TFinance')
print(f'Structure Shift Budget: {args.ood_budget}')
print(f'Average AUROC: {logger.print_statistics()[0]:.4f} ± {logger.print_statistics()[1]:.4f}')
print(f'Average AUPR: {logger.print_statistics()[2]:.4f} ± {logger.print_statistics()[3]:.4f}')
print(f'Average Time: {np.mean(durations):.2f}s ± {np.std(durations):.2f}s')
print('=' * 50)