import numpy as np
import pickle
import os

def generate_mock_logits(num_nodes, num_classes, seed=42):
    """Generate mock logits for testing without PyTorch"""
    print(f"Generating mock logits: {num_nodes} nodes, {num_classes} classes")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate realistic logits
    logits = np.random.randn(num_nodes, num_classes)
    
    # Add some structure to make it more realistic
    # Make the logits slightly biased towards random classes
    for i in range(num_nodes):
        # Add some bias towards a random class (simulating model predictions)
        bias_class = np.random.randint(0, num_classes)
        logits[i, bias_class] += 2.0  # Add bias
    
    # Normalize to make it more realistic
    logits = logits - logits.mean(axis=1, keepdims=True)
    
    return logits

def generate_multiple_runs_logits(num_nodes, num_classes, num_runs=5):
    """Generate logits for multiple runs"""
    logits_list = []
    
    for run in range(num_runs):
        print(f"Generating logits for run {run+1}/{num_runs}")
        logits = generate_mock_logits(num_nodes, num_classes, seed=42 + run)
        logits_list.append(logits)
    
    return logits_list

def save_logits_as_pickle(logits, filename):
    """Save logits as pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump(logits, f)
    print(f"Saved logits to: {filename}")

def main():
    # TFinance dataset typical parameters
    # These are typical values for TFinance dataset
    num_nodes = 10000  # Typical size for TFinance
    num_classes = 2     # Binary classification for TFinance
    
    print("Generating test logits for TFinance dataset...")
    print(f"Dataset parameters: {num_nodes} nodes, {num_classes} classes")
    
    # Generate single run logits
    print("\nGenerating single run logits...")
    single_logits = generate_mock_logits(num_nodes, num_classes)
    save_logits_as_pickle(single_logits, 'test_logits_single.pkl')
    print(f"Logits shape: {single_logits.shape}")
    
    # Generate multiple runs logits
    print("\nGenerating multiple runs logits...")
    multiple_logits = generate_multiple_runs_logits(num_nodes, num_classes, num_runs=5)
    save_logits_as_pickle(multiple_logits, 'test_logits_multiple.pkl')
    print(f"Number of runs: {len(multiple_logits)}")
    print(f"Each logits shape: {multiple_logits[0].shape}")
    
    # Test loading
    print(f"\nTesting logits loading...")
    with open('test_logits_single.pkl', 'rb') as f:
        loaded_single = pickle.load(f)
    with open('test_logits_multiple.pkl', 'rb') as f:
        loaded_multiple = pickle.load(f)
    
    print(f"Single logits loaded successfully: {loaded_single.shape}")
    print(f"Multiple logits loaded successfully: {len(loaded_multiple)} runs")
    
    # Show some statistics
    print(f"\nLogits statistics:")
    print(f"  Single logits - Mean: {loaded_single.mean():.4f}, Std: {loaded_single.std():.4f}")
    print(f"  Single logits - Min: {loaded_single.min():.4f}, Max: {loaded_single.max():.4f}")
    
    print(f"\nTest logits generation completed!")
    print(f"\nYou can now use these files to test your OOD detection script:")
    print(f"  python test_ood_tfinance.py --logits_path test_logits_single.pkl --ood GRASP")
    print(f"  python test_ood_tfinance.py --logits_path test_logits_multiple.pkl --ood MSP")
    
    # Also create a .pt version for compatibility
    print(f"\nCreating .pt version for PyTorch compatibility...")
    try:
        import torch
        torch_single = torch.from_numpy(single_logits).float()
        torch_multiple = [torch.from_numpy(logits).float() for logits in multiple_logits]
        
        torch.save(torch_single, 'test_logits_single.pt')
        torch.save(torch_multiple, 'test_logits_multiple.pt')
        
        print(f"PyTorch .pt files created successfully!")
        print(f"  test_logits_single.pt")
        print(f"  test_logits_multiple.pt")
        
    except ImportError:
        print(f"PyTorch not available, only pickle files created")

if __name__ == "__main__":
    main()