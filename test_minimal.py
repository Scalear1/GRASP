#!/usr/bin/env python3

import random

def load_logits_simple(filename):
    """Load logits from text file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    header = lines[0].strip().split()
    num_nodes, num_classes = int(header[0]), int(header[1])
    
    # Parse logits
    logits = []
    for i in range(1, len(lines)):
        if lines[i].strip():
            node_logits = [float(x) for x in lines[i].strip().split()]
            logits.append(node_logits)
    
    print(f"Loaded logits: {len(logits)} nodes x {len(logits[0])} classes")
    return logits

def main():
    # Load logits
    logits = load_logits_simple('test_logits_single.txt')
    
    # Create simple test
    test_nodes = [100, 200, 300, 400, 500]
    ood_nodes = [600, 700, 800, 900, 1000]
    
    print(f"Test nodes: {test_nodes}")
    print(f"OOD nodes: {ood_nodes}")
    
    # Get scores
    scores = []
    for node_logits in logits:
        scores.append(max(node_logits))
    
    print(f"Scores length: {len(scores)}")
    print(f"Max score index: {len(scores) - 1}")
    
    # Test indexing
    try:
        test_id_scores = [scores[i] for i in test_nodes]
        test_ood_scores = [scores[i] for i in ood_nodes]
        print(f"Test ID scores: {test_id_scores}")
        print(f"Test OOD scores: {test_ood_scores}")
        print("Indexing successful!")
    except IndexError as e:
        print(f"IndexError: {e}")
        print(f"Tried to access indices: {test_nodes + ood_nodes}")
        print(f"Max index needed: {max(test_nodes + ood_nodes)}")
        print(f"Available indices: 0 to {len(scores) - 1}")

if __name__ == "__main__":
    main()