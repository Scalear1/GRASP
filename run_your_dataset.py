#!/usr/bin/env python3
"""
使用您的DGL数据集运行GRASP OOD检测的示例脚本
"""

import os
import sys
import argparse

def main():
    # 设置基本参数
    dataset_name = "your_dgl_dataset"  # 请修改为您的数据集名称
    method = "gcn"  # 可以选择: gcn, gat, sage, etc.
    device = 0
    runs = 5
    
    print(f"Running GRASP OOD detection on {dataset_name} dataset")
    print(f"Method: {method}")
    print(f"Device: {device}")
    print(f"Runs: {runs}")
    
    # 第一步：训练基础模型（如果还没有预训练模型）
    print("\n=== Step 1: Training base model ===")
    train_cmd = f"python train_id.py --dataset {dataset_name} --method {method} --device {device} --runs {runs}"
    print(f"Command: {train_cmd}")
    # os.system(train_cmd)  # 取消注释以实际运行
    
    # 第二步：运行GRASP OOD检测
    print("\n=== Step 2: Running GRASP OOD detection ===")
    test_cmd = f"python test_ood.py --dataset {dataset_name} --method {method} --ood GRASP --device {device} --runs {runs}"
    print(f"Command: {test_cmd}")
    # os.system(test_cmd)  # 取消注释以实际运行
    
    # 第三步：运行其他baseline方法进行对比
    print("\n=== Step 3: Running baseline methods ===")
    baseline_methods = ['MSP', 'Energy', 'KNN', 'ODIN', 'Mahalanobis']
    
    for ood_method in baseline_methods:
        baseline_cmd = f"python test_ood.py --dataset {dataset_name} --method {method} --ood {ood_method} --device {device} --runs {runs}"
        print(f"Command: {baseline_cmd}")
        # os.system(baseline_cmd)  # 取消注释以实际运行
    
    print("\n=== All experiments completed! ===")
    print(f"Results will be saved in: results/{dataset_name}-{method}.csv")

if __name__ == "__main__":
    main()