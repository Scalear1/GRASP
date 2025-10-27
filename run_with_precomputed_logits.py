#!/usr/bin/env python3
"""
使用预计算logits运行GRASP OOD检测的示例脚本
"""

import os
import sys

def main():
    dataset_name = "your_dataset_name"  # 修改为您的数据集名称
    method = "gcn"
    device = 0
    runs = 5
    
    print(f"🚀 Running GRASP OOD detection with precomputed logits")
    print(f"Dataset: {dataset_name}")
    print(f"Method: {method}")
    print(f"Device: {device}")
    print(f"Runs: {runs}")
    
    # 方式1: 使用单个logits文件（推荐，如果所有runs使用相同的logits）
    single_logits_file = f"logits/{dataset_name}_logits.pt"
    if os.path.exists(single_logits_file):
        print(f"\n=== 使用单个logits文件 ===")
        cmd = f"python test_ood.py --dataset {dataset_name} --method {method} --ood GRASP --device {device} --runs {runs} --logits_file {single_logits_file} --force_logits"
        print(f"Command: {cmd}")
        os.system(cmd)
    else:
        print(f"\n=== 使用多个logits文件（按run编号） ===")
        # 方式2: 使用logits目录（每个run有单独的logits文件）
        logits_dir = f"logits/{dataset_name}/{method}"
        
        # 检查logits文件是否存在
        missing_files = []
        for run in range(runs):
            logit_file = f"{logits_dir}/logit{run}.pt"
            if not os.path.exists(logit_file):
                missing_files.append(logit_file)
        
        if missing_files:
            print(f"❌ Missing logits files:")
            for f in missing_files:
                print(f"   {f}")
            print(f"\n💡 请确保您的logits文件存在于以下位置之一:")
            print(f"   1. 单个文件: {single_logits_file}")
            print(f"   2. 多个文件: {logits_dir}/logit0.pt, logit1.pt, ...")
            return
        
        cmd = f"python test_ood.py --dataset {dataset_name} --method {method} --ood GRASP --device {device} --runs {runs} --logits_dir {logits_dir} --force_logits"
        print(f"Command: {cmd}")
        os.system(cmd)
    
    # 运行其他baseline方法
    print(f"\n=== 运行其他baseline方法 ===")
    baseline_methods = ['MSP', 'Energy', 'KNN', 'ODIN']
    
    for ood_method in baseline_methods:
        print(f"\n--- Running {ood_method} ---")
        if os.path.exists(single_logits_file):
            cmd = f"python test_ood.py --dataset {dataset_name} --method {method} --ood {ood_method} --device {device} --runs {runs} --logits_file {single_logits_file} --force_logits"
        else:
            cmd = f"python test_ood.py --dataset {dataset_name} --method {method} --ood {ood_method} --device {device} --runs {runs} --logits_dir {logits_dir} --force_logits"
        print(f"Command: {cmd}")
        os.system(cmd)
    
    print(f"\n✅ All experiments completed!")
    print(f"Results saved in: results/{dataset_name}-{method}.csv")

if __name__ == "__main__":
    main()