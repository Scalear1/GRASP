import argparse
import sys
import os
import time
import numpy as np
import random
import struct

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def load_logits_from_text(filename):
    """Load logits from text file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    if len(lines) == 1:
        # Single run format
        header = lines[0].strip().split()
        num_nodes, num_classes = int(header[0]), int(header[1])
        logits = []
        for i in range(1, len(lines)):
            if lines[i].strip():  # Skip empty lines
                node_logits = [float(x) for x in lines[i].strip().split()]
                logits.append(node_logits)
        return np.array(logits)
    else:
        # Multiple runs format
        num_runs = int(lines[0].strip())
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
            
            logits_list.append(np.array(run_logits))
        
        return logits_list

def load_logits_from_binary(filename):
    """Load logits from binary file"""
    with open(filename, 'rb') as f:
        # Read header
        header = f.read(8)  # 2 integers
        num_nodes, num_classes = struct.unpack('ii', header)
        
        # Read data
        logits = []
        for i in range(num_nodes):
            node_logits = []
            for j in range(num_classes):
                value = struct.unpack('f', f.read(4))[0]
                node_logits.append(value)
            logits.append(node_logits)
    
    return np.array(logits)

def create_mock_tfinance_data():
    """Create mock TFinance dataset for testing"""
    print("Creating mock TFinance dataset...")
    
    # Mock dataset parameters
    num_nodes = 10000
    num_features = 64
    num_classes = 2
    
    # Generate mock data
    np.random.seed(42)
    
    # Node features
    node_features = np.random.randn(num_nodes, num_features)
    
    # Node labels (binary)
    node_labels = np.random.randint(0, num_classes, num_nodes)
    
    # Edge index (sparse graph)
    num_edges = num_nodes * 10  # Average degree of 10
    edge_index = []
    for i in range(num_edges):
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if src != dst:
            edge_index.append([src, dst])
    
    edge_index = np.array(edge_index).T
    
    print(f"Mock dataset created:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {edge_index.shape[1]}")
    print(f"  Features: {num_features}")
    print(f"  Classes: {num_classes}")
    
    return {
        'num_nodes': num_nodes,
        'node_features': node_features,
        'node_labels': node_labels,
        'edge_index': edge_index
    }

def structure_shift_dataset_mock(data, run, args, ood_budget_per_graph=0.1):
    """Create structure shift OOD dataset for mock data"""
    seed = run + args.seed
    np.random.seed(seed)
    
    num_nodes = data['num_nodes']
    node_labels = data['node_labels']
    edge_index = data['edge_index']
    
    # Extract nodes belonging to each class
    num_classes = len(np.unique(node_labels))
    nodes_per_class = []
    for c in range(num_classes):
        nodes_per_class.append(np.where(node_labels == c)[0])
    
    # Create OOD data by perturbing edges
    ood_nodes = []
    ood_edges = []
    
    for c in range(num_classes):
        # Find intra-community edges
        class_nodes = nodes_per_class[c]
        intra_edges = []
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src in class_nodes and dst in class_nodes:
                intra_edges.append(i)
        
        # Calculate budget
        budget = int(ood_budget_per_graph * len(intra_edges))
        
        # Sample edges to perturb
        if len(intra_edges) > 0:
            perturbed_edges = np.random.choice(intra_edges, min(budget, len(intra_edges)), replace=False)
            
            for edge_idx in perturbed_edges:
                src, dst = edge_index[0, edge_idx], edge_index[1, edge_idx]
                ood_nodes.extend([src, dst])
                
                # Create cross-community edge
                other_class = (c + 1) % num_classes
                other_nodes = nodes_per_class[other_class]
                if len(other_nodes) > 0:
                    new_dst = np.random.choice(other_nodes)
                    ood_edges.append([src, new_dst])
    
    # Create OOD dataset
    ood_nodes = np.unique(ood_nodes)
    ood_edges = np.array(ood_edges).T if ood_edges else np.array([[], []])
    
    # Create ID dataset (remaining nodes)
    id_nodes = np.setdiff1d(np.arange(num_nodes), ood_nodes)
    
    # Split ID dataset
    train_size = int(0.6 * len(id_nodes))
    valid_size = int(0.2 * len(id_nodes))
    
    np.random.shuffle(id_nodes)
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
    all_scores = np.concatenate([id_scores, ood_scores])
    all_labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    
    # Calculate AUROC (simplified)
    sorted_indices = np.argsort(all_scores)[::-1]  # Sort by score descending
    sorted_labels = all_labels[sorted_indices]
    
    # Calculate TPR and FPR
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    
    # Normalize
    total_pos = np.sum(sorted_labels)
    total_neg = len(sorted_labels) - total_pos
    
    if total_pos > 0 and total_neg > 0:
        tpr = tp / total_pos
        fpr = fp / total_neg
        
        # Calculate AUROC (trapezoidal rule)
        auroc = np.trapz(tpr, fpr)
        
        # Calculate AUPR (simplified)
        precision = tp / (tp + fp + 1e-8)
        recall = tpr
        aupr = np.trapz(precision, recall)
    else:
        auroc = 0.5
        aupr = 0.5
    
    return auroc, aupr

def main():
    parser = argparse.ArgumentParser(description='Simple OOD Detection on TFinance Dataset')
    parser.add_argument('--logits_path', type=str, required=True, help='Path to logits file')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--ood_budget', type=float, default=0.1, help='OOD budget for structure shift')
    
    args = parser.parse_args()
    
    print(f"Arguments: {args}")
    
    # Load logits
    print(f"Loading logits from {args.logits_path}")
    if args.logits_path.endswith('.txt'):
        logits = load_logits_from_text(args.logits_path)
    elif args.logits_path.endswith('.bin'):
        logits = load_logits_from_binary(args.logits_path)
    else:
        raise ValueError("Unsupported logits file format")
    
    print(f"Logits loaded: {logits.shape if hasattr(logits, 'shape') else f'list of {len(logits)} arrays'}")
    
    # Create mock dataset
    data = create_mock_tfinance_data()
    
    # Testing
    results = []
    durations = []
    
    for run in range(args.runs):
        t = time.time()
        print(f'----Run {run+1}/{args.runs}')
        set_seed(run + args.seed)
        
        # Create structure shift dataset
        dataset_split = structure_shift_dataset_mock(data, run, args, args.ood_budget)
        
        # Get logits for current run
        if isinstance(logits, list) and len(logits) > run:
            current_logits = logits[run]
        else:
            current_logits = logits
        
        # Get scores (using MSP method - max softmax probability)
        scores = current_logits.max(axis=1)  # Simple MSP implementation
        
        # Evaluate OOD detection
        test_id_scores = scores[dataset_split['test_nodes']]
        test_ood_scores = scores[dataset_split['ood_nodes']]
        
        auroc, aupr = evaluate_ood_simple(test_id_scores, test_ood_scores)
        
        duration = time.time() - t
        durations.append(duration)
        
        print(f'Run {run+1} completed in {duration:.2f}s')
        print(f'AUROC: {auroc:.4f}, AUPR: {aupr:.4f}')
        
        results.append((auroc, aupr))
    
    # Print final results
    aurocs = [r[0] for r in results]
    auprs = [r[1] for r in results]
    
    print('=' * 50)
    print('Final Results:')
    print(f'Dataset: TFinance (Mock)')
    print(f'Structure Shift Budget: {args.ood_budget}')
    print(f'Average AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}')
    print(f'Average AUPR: {np.mean(auprs):.4f} ± {np.std(auprs):.4f}')
    print(f'Average Time: {np.mean(durations):.2f}s ± {np.std(durations):.2f}s')
    print('=' * 50)

if __name__ == "__main__":
    main()