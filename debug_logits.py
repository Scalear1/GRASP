#!/usr/bin/env python3

def load_logits_debug(filename):
    """Debug logits loading"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    print(f"Total lines: {len(lines)}")
    print(f"First line: {lines[0].strip()}")
    
    # Parse header
    header = lines[0].strip().split()
    num_nodes, num_classes = int(header[0]), int(header[1])
    print(f"Header: {num_nodes} nodes, {num_classes} classes")
    
    # Parse logits
    logits = []
    for i in range(1, len(lines)):
        if lines[i].strip():
            node_logits = [float(x) for x in lines[i].strip().split()]
            logits.append(node_logits)
    
    print(f"Loaded logits: {len(logits)} nodes")
    print(f"First node logits: {logits[0]}")
    print(f"Last node logits: {logits[-1]}")
    
    return logits

if __name__ == "__main__":
    logits = load_logits_debug('test_logits_single.txt')
    print(f"Logits length: {len(logits)}")
    print(f"Each node has {len(logits[0])} classes")