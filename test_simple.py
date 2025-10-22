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

def create_mock_data():
    """Create mock dataset"""
    num_nodes = 10000
    node_labels = [random.randint(0, 1) for _ in range(num_nodes)]
    
    # Create simple split
    all_nodes = list(range(num_nodes))
    random.shuffle(all_nodes)
    
    train_size = int(0.6 * num_nodes)
    valid_size = int(0.2 * num_nodes)
    
    train_nodes = all_nodes[:train_size]
    valid_nodes = all_nodes[train_size:train_size + valid_size]
    test_nodes = all_nodes[train_size + valid_size:]
    
    # Create OOD nodes (simulate structure shift)
    ood_nodes = random.sample(all_nodes, int(0.3 * num_nodes))
    
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
    parser = argparse.ArgumentParser(description='Simple OOD Detection Test')
    parser.add_argument('--logits_path', type=str, required=True, help='Path to logits file')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Testing OOD detection with logits: {args.logits_path}")
    
    # Load logits
    logits = load_logits_simple(args.logits_path)
    
    # Create mock data
    data = create_mock_data()
    
    results = []
    
    for run in range(args.runs):
        print(f"\n----Run {run+1}/{args.runs}")
        random.seed(args.seed + run)
        
        # Get logits for current run
        if isinstance(logits, list) and len(logits) > 0 and isinstance(logits[0], list) and len(logits) > run:
            # Multiple runs format
            current_logits = logits[run]
        else:
            # Single run format
            current_logits = logits
        
        # Get scores (using MSP method - max logits)
        scores = []
        for node_logits in current_logits:
            scores.append(max(node_logits))
        
        # Evaluate OOD detection
        test_id_scores = [scores[i] for i in data['test_nodes']]
        test_ood_scores = [scores[i] for i in data['ood_nodes']]
        
        auroc, aupr = evaluate_ood_simple(test_id_scores, test_ood_scores)
        
        print(f'AUROC: {auroc:.4f}, AUPR: {aupr:.4f}')
        results.append((auroc, aupr))
    
    # Print final results
    aurocs = [r[0] for r in results]
    auprs = [r[1] for r in results]
    
    avg_auroc = sum(aurocs) / len(aurocs)
    avg_aupr = sum(auprs) / len(auprs)
    
    print('\n' + '=' * 50)
    print('Final Results:')
    print(f'Average AUROC: {avg_auroc:.4f}')
    print(f'Average AUPR: {avg_aupr:.4f}')
    print('=' * 50)

if __name__ == "__main__":
    main()