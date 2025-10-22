#!/usr/bin/env python3

import argparse
import random
import time

def load_logits_simple(filename):
    """Load logits from text file - simplified version"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Check if it's single run or multiple runs format
    first_line = lines[0].strip().split()
    
    if len(first_line) == 2:
        # Single run format: "num_nodes num_classes"
        num_nodes, num_classes = int(first_line[0]), int(first_line[1])
        
        # Parse logits
        logits = []
        for i in range(1, len(lines)):
            if lines[i].strip():
                node_logits = [float(x) for x in lines[i].strip().split()]
                logits.append(node_logits)
        
        print(f"Loaded single run logits: {len(logits)} nodes x {len(logits[0])} classes")
        return logits
    else:
        # Multiple runs format: first line is number of runs
        num_runs = int(first_line[0])
        logits_list = []
        line_idx = 1
        
        for run in range(num_runs):
            header = lines[line_idx].strip().split()
            num_nodes, num_classes = int(header[0]), int(header[1])
            line_idx += 1
            
            run_logits = []
            for i in range(num_nodes):
                node_logits = [float(x) for x in lines[line_idx].strip().split()]
                run_logits.append(node_logits)
                line_idx += 1
            
            logits_list.append(run_logits)
        
        print(f"Loaded multiple runs logits: {len(logits_list)} runs, each {len(logits_list[0])} nodes x {len(logits_list[0][0])} classes")
        return logits_list

def create_mock_tfinance_data():
    """Create mock TFinance dataset"""
    num_nodes = 10000
    num_classes = 2
    
    # Generate node labels (binary classification)
    random.seed(42)
    node_labels = [random.randint(0, num_classes-1) for _ in range(num_nodes)]
    
    # Generate edge index (sparse graph)
    num_edges = num_nodes * 10  # Average degree of 10
    edge_index = []
    for i in range(num_edges):
        src = random.randint(0, num_nodes-1)
        dst = random.randint(0, num_nodes-1)
        if src != dst:
            edge_index.append([src, dst])
    
    print(f"Mock TFinance dataset created:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {len(edge_index)}")
    print(f"  Classes: {num_classes}")
    
    return {
        'num_nodes': num_nodes,
        'node_labels': node_labels,
        'edge_index': edge_index
    }

def structure_shift_dataset_mock(data, run, args, ood_budget_per_graph=0.1):
    """Create structure shift OOD dataset for mock data"""
    seed = run + args.seed
    random.seed(seed)
    
    num_nodes = data['num_nodes']
    node_labels = data['node_labels']
    edge_index = data['edge_index']
    
    # Extract nodes belonging to each class
    nodes_per_class = [[], []]
    for i, label in enumerate(node_labels):
        nodes_per_class[label].append(i)
    
    # Create OOD data by perturbing edges
    ood_nodes = set()
    
    for c in range(2):  # Binary classification
        # Find intra-community edges
        class_nodes = set(nodes_per_class[c])
        intra_edges = []
        
        for i, (src, dst) in enumerate(edge_index):
            if src in class_nodes and dst in class_nodes:
                intra_edges.append(i)
        
        # Calculate budget
        budget = int(ood_budget_per_graph * len(intra_edges))
        
        # Sample edges to perturb
        if len(intra_edges) > 0:
            perturbed_edges = random.sample(intra_edges, min(budget, len(intra_edges)))
            
            for edge_idx in perturbed_edges:
                src, dst = edge_index[edge_idx]
                ood_nodes.add(src)
                ood_nodes.add(dst)
    
    # Create ID dataset (remaining nodes)
    all_nodes = set(range(num_nodes))
    id_nodes = list(all_nodes - ood_nodes)
    ood_nodes = list(ood_nodes)
    
    # Split ID dataset
    train_size = int(0.6 * len(id_nodes))
    valid_size = int(0.2 * len(id_nodes))
    
    random.shuffle(id_nodes)
    train_nodes = id_nodes[:train_size]
    valid_nodes = id_nodes[train_size:train_size + valid_size]
    test_nodes = id_nodes[train_size + valid_size:]
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_nodes)} nodes")
    print(f"  Valid: {len(valid_nodes)} nodes")
    print(f"  Test ID: {len(test_nodes)} nodes")
    print(f"  OOD: {len(ood_nodes)} nodes")
    
    return {
        'train_nodes': train_nodes,
        'valid_nodes': valid_nodes,
        'test_nodes': test_nodes,
        'ood_nodes': ood_nodes,
        'num_nodes': num_nodes
    }

def evaluate_ood_simple(id_scores, ood_scores):
    """Simple OOD evaluation"""
    # Combine scores and create labels
    all_scores = id_scores + ood_scores
    all_labels = [0] * len(id_scores) + [1] * len(ood_scores)
    
    # Sort by score descending
    combined = list(zip(all_scores, all_labels))
    combined.sort(key=lambda x: x[0], reverse=True)
    
    # Calculate AUROC (simplified)
    tp = 0
    fp = 0
    total_pos = sum(all_labels)
    total_neg = len(all_labels) - total_pos
    
    if total_pos == 0 or total_neg == 0:
        return 0.5, 0.5
    
    # Calculate AUROC
    tpr_values = []
    fpr_values = []
    
    for score, label in combined:
        if label == 1:
            tp += 1
        else:
            fp += 1
        
        tpr = tp / total_pos
        fpr = fp / total_neg
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    
    # Calculate AUROC using trapezoidal rule
    auroc = 0
    for i in range(1, len(fpr_values)):
        auroc += (fpr_values[i] - fpr_values[i-1]) * (tpr_values[i] + tpr_values[i-1]) / 2
    
    # Calculate AUPR (simplified)
    precision_values = []
    recall_values = []
    
    tp = 0
    fp = 0
    for score, label in combined:
        if label == 1:
            tp += 1
        else:
            fp += 1
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
            recall = tp / total_pos
            precision_values.append(precision)
            recall_values.append(recall)
    
    # Calculate AUPR using trapezoidal rule
    aupr = 0
    for i in range(1, len(recall_values)):
        aupr += (recall_values[i] - recall_values[i-1]) * (precision_values[i] + precision_values[i-1]) / 2
    
    return auroc, aupr

def main():
    parser = argparse.ArgumentParser(description='Final OOD Detection on TFinance Dataset')
    parser.add_argument('--logits_path', type=str, required=True, help='Path to logits file')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--ood_budget', type=float, default=0.1, help='OOD budget for structure shift')
    
    args = parser.parse_args()
    
    print(f"Arguments: {args}")
    
    # Load logits
    print(f"Loading logits from {args.logits_path}")
    logits = load_logits_simple(args.logits_path)
    
    # Create mock dataset
    data = create_mock_tfinance_data()
    
    # Testing
    results = []
    durations = []
    
    for run in range(args.runs):
        t = time.time()
        print(f'\n----Run {run+1}/{args.runs}')
        random.seed(run + args.seed)
        
        # Create structure shift dataset
        dataset_split = structure_shift_dataset_mock(data, run, args, args.ood_budget)
        
        # Get logits for current run
        # Check if logits is a list of runs (multiple runs format)
        if isinstance(logits, list) and len(logits) > 0 and isinstance(logits[0], list) and len(logits[0]) > 0 and isinstance(logits[0][0], list):
            # Multiple runs format - logits is a list of runs
            current_logits = logits[run]
        else:
            # Single run format - logits is already the correct format
            current_logits = logits
        
        # Get scores (using MSP method - max logits)
        scores = []
        for node_logits in current_logits:
            if isinstance(node_logits, list):
                scores.append(max(node_logits))
            else:
                # If node_logits is a single float, use it directly
                scores.append(node_logits)
        
        # Evaluate OOD detection
        test_id_scores = [scores[i] for i in dataset_split['test_nodes']]
        test_ood_scores = [scores[i] for i in dataset_split['ood_nodes']]
        
        auroc, aupr = evaluate_ood_simple(test_id_scores, test_ood_scores)
        
        duration = time.time() - t
        durations.append(duration)
        
        print(f'Run {run+1} completed in {duration:.2f}s')
        print(f'AUROC: {auroc:.4f}, AUPR: {aupr:.4f}')
        
        results.append((auroc, aupr))
    
    # Print final results
    aurocs = [r[0] for r in results]
    auprs = [r[1] for r in results]
    
    avg_auroc = sum(aurocs) / len(aurocs)
    avg_aupr = sum(auprs) / len(auprs)
    avg_time = sum(durations) / len(durations)
    
    # Calculate standard deviations
    auroc_var = sum((x - avg_auroc) ** 2 for x in aurocs) / len(aurocs)
    aupr_var = sum((x - avg_aupr) ** 2 for x in auprs) / len(auprs)
    time_var = sum((x - avg_time) ** 2 for x in durations) / len(durations)
    
    auroc_std = auroc_var ** 0.5
    aupr_std = aupr_var ** 0.5
    time_std = time_var ** 0.5
    
    print('\n' + '=' * 50)
    print('Final Results:')
    print(f'Dataset: TFinance (Mock)')
    print(f'Structure Shift Budget: {args.ood_budget}')
    print(f'Average AUROC: {avg_auroc:.4f} ± {auroc_std:.4f}')
    print(f'Average AUPR: {avg_aupr:.4f} ± {aupr_std:.4f}')
    print(f'Average Time: {avg_time:.2f}s ± {time_std:.2f}s')
    print('=' * 50)

if __name__ == "__main__":
    main()