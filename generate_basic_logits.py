import random
import struct
import os

def generate_random_logits(num_nodes, num_classes, seed=42):
    """Generate random logits using only standard library"""
    print(f"Generating logits: {num_nodes} nodes, {num_classes} classes")
    
    # Set random seed
    random.seed(seed)
    
    # Generate logits as list of lists
    logits = []
    for i in range(num_nodes):
        node_logits = []
        for j in range(num_classes):
            # Generate random float between -3 and 3
            value = random.uniform(-3.0, 3.0)
            # Add some bias for one class (simulating model predictions)
            if j == random.randint(0, num_classes-1):
                value += 2.0
            node_logits.append(value)
        logits.append(node_logits)
    
    return logits

def save_logits_as_text(logits, filename):
    """Save logits as simple text file"""
    with open(filename, 'w') as f:
        f.write(f"{len(logits)} {len(logits[0])}\n")  # Header: num_nodes num_classes
        for node_logits in logits:
            f.write(" ".join(f"{x:.6f}" for x in node_logits) + "\n")
    print(f"Saved logits to: {filename}")

def save_logits_as_binary(logits, filename):
    """Save logits as binary file"""
    with open(filename, 'wb') as f:
        # Write header
        num_nodes = len(logits)
        num_classes = len(logits[0])
        f.write(struct.pack('ii', num_nodes, num_classes))
        
        # Write data
        for node_logits in logits:
            for value in node_logits:
                f.write(struct.pack('f', value))
    print(f"Saved logits to: {filename}")

def main():
    # TFinance dataset typical parameters
    num_nodes = 10000  # Typical size for TFinance
    num_classes = 2     # Binary classification for TFinance
    
    print("Generating test logits for TFinance dataset...")
    print(f"Dataset parameters: {num_nodes} nodes, {num_classes} classes")
    
    # Generate single run logits
    print("\nGenerating single run logits...")
    single_logits = generate_random_logits(num_nodes, num_classes, seed=42)
    
    # Save in different formats
    save_logits_as_text(single_logits, 'test_logits_single.txt')
    save_logits_as_binary(single_logits, 'test_logits_single.bin')
    
    print(f"Logits shape: {len(single_logits)} x {len(single_logits[0])}")
    
    # Generate multiple runs logits
    print("\nGenerating multiple runs logits...")
    multiple_logits = []
    for run in range(5):
        print(f"Generating logits for run {run+1}/5")
        logits = generate_random_logits(num_nodes, num_classes, seed=42 + run)
        multiple_logits.append(logits)
    
    # Save multiple runs
    with open('test_logits_multiple.txt', 'w') as f:
        f.write(f"{len(multiple_logits)}\n")  # Number of runs
        for run_logits in multiple_logits:
            f.write(f"{len(run_logits)} {len(run_logits[0])}\n")
            for node_logits in run_logits:
                f.write(" ".join(f"{x:.6f}" for x in node_logits) + "\n")
    
    print(f"Saved multiple runs logits to: test_logits_multiple.txt")
    print(f"Number of runs: {len(multiple_logits)}")
    
    # Show some statistics
    print(f"\nLogits statistics:")
    all_values = []
    for node_logits in single_logits:
        all_values.extend(node_logits)
    
    mean_val = sum(all_values) / len(all_values)
    min_val = min(all_values)
    max_val = max(all_values)
    
    print(f"  Single logits - Mean: {mean_val:.4f}")
    print(f"  Single logits - Min: {min_val:.4f}, Max: {max_val:.4f}")
    
    print(f"\nTest logits generation completed!")
    print(f"\nGenerated files:")
    print(f"  test_logits_single.txt - Text format")
    print(f"  test_logits_single.bin - Binary format")
    print(f"  test_logits_multiple.txt - Multiple runs")
    
    print(f"\nYou can now use these files to test your OOD detection script.")
    print(f"Note: You may need to modify the loading code in test_ood_tfinance.py")
    print(f"to handle these text/binary formats instead of .pt files.")

if __name__ == "__main__":
    main()