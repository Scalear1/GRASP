import torch
import dgl
from dgl.data import TFinancDataset
import numpy as np
import os

def load_tfinance_data():
    """Load tfinance dataset and get basic info"""
    # Load DGL graph
    graph = dgl.data.TFinancDataset()[0]
    
    # Extract features and labels
    node_feats = graph.ndata['feature']
    node_labels = graph.ndata['label']
    edge_index = graph.edges()
    
    print(f"TFinance dataset info:")
    print(f"  Number of nodes: {graph.num_nodes()}")
    print(f"  Number of edges: {graph.num_edges()}")
    print(f"  Feature dimension: {node_feats.shape[1]}")
    print(f"  Number of classes: {node_labels.max().item() + 1}")
    print(f"  Label distribution: {torch.bincount(node_labels)}")
    
    return graph, node_feats, node_labels

def generate_mock_logits(num_nodes, num_classes, device='cpu'):
    """Generate mock logits for testing"""
    print(f"Generating mock logits: {num_nodes} nodes, {num_classes} classes")
    
    # Generate realistic logits using a simple model
    torch.manual_seed(42)  # For reproducibility
    
    # Create a simple "model" that generates logits
    # This simulates what a trained GNN would output
    logits = torch.randn(num_nodes, num_classes, device=device)
    
    # Add some structure to make it more realistic
    # Make the logits slightly biased towards the correct class
    for i in range(num_nodes):
        # Add some bias towards a random class (simulating model predictions)
        bias_class = torch.randint(0, num_classes, (1,)).item()
        logits[i, bias_class] += 2.0  # Add bias
    
    # Normalize to make it more realistic
    logits = logits - logits.mean(dim=1, keepdim=True)
    
    return logits

def generate_multiple_runs_logits(num_nodes, num_classes, num_runs=5, device='cpu'):
    """Generate logits for multiple runs"""
    logits_list = []
    
    for run in range(num_runs):
        print(f"Generating logits for run {run+1}/{num_runs}")
        torch.manual_seed(42 + run)  # Different seed for each run
        logits = generate_mock_logits(num_nodes, num_classes, device)
        logits_list.append(logits)
    
    return logits_list

def main():
    print("Loading TFinance dataset...")
    graph, node_feats, node_labels = load_tfinance_data()
    
    num_nodes = graph.num_nodes()
    num_classes = node_labels.max().item() + 1
    
    print(f"\nGenerating test logits...")
    
    # Generate single run logits
    single_logits = generate_mock_logits(num_nodes, num_classes)
    torch.save(single_logits, 'test_logits_single.pt')
    print(f"Saved single run logits to: test_logits_single.pt")
    print(f"Logits shape: {single_logits.shape}")
    
    # Generate multiple runs logits
    multiple_logits = generate_multiple_runs_logits(num_nodes, num_classes, num_runs=5)
    torch.save(multiple_logits, 'test_logits_multiple.pt')
    print(f"Saved multiple runs logits to: test_logits_multiple.pt")
    print(f"Number of runs: {len(multiple_logits)}")
    print(f"Each logits shape: {multiple_logits[0].shape}")
    
    # Test loading
    print(f"\nTesting logits loading...")
    loaded_single = torch.load('test_logits_single.pt')
    loaded_multiple = torch.load('test_logits_multiple.pt')
    
    print(f"Single logits loaded successfully: {loaded_single.shape}")
    print(f"Multiple logits loaded successfully: {len(loaded_multiple)} runs")
    
    # Show some statistics
    print(f"\nLogits statistics:")
    print(f"  Single logits - Mean: {loaded_single.mean():.4f}, Std: {loaded_single.std():.4f}")
    print(f"  Single logits - Min: {loaded_single.min():.4f}, Max: {loaded_single.max():.4f}")
    
    print(f"\nTest logits generation completed!")
    print(f"You can now use these files to test your OOD detection script:")
    print(f"  python test_ood_tfinance.py --logits_path test_logits_single.pt --ood GRASP")
    print(f"  python test_ood_tfinance.py --logits_path test_logits_multiple.pt --ood MSP")

if __name__ == "__main__":
    main()