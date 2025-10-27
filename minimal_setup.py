#!/usr/bin/env python3
"""
检查运行test_ood_logits_only.py所需的最小文件集合
"""

print("运行 test_ood_logits_only.py 需要以下文件：")
print("\n=== 核心文件 (必需) ===")
required_files = [
    "test_ood_logits_only.py",    # 主脚本
    "grasp.py",                   # GRASP算法实现
    "baselines.py",               # 其他OOD检测方法
    "dataset.py",                 # 数据集加载
    "data_utils.py",              # 数据处理工具
    "logger.py",                  # 结果记录
    "parse.py",                   # 参数解析
    "hyparams.py",                # 超参数配置 (可选)
    "load_data.py",               # 数据加载工具
]

for f in required_files:
    print(f"  ✓ {f}")

print("\n=== 目录结构 ===")
print("  ✓ results/          # 结果保存目录 (会自动创建)")
print("  ✓ logits/           # logits文件目录")

print("\n=== 您的数据相关 ===")
print("  ✓ 您的logits文件    # 例: logits/your_dataset_logits.pt")
print("  ✓ 在dataset.py中添加您的数据集加载函数")

print("\n=== 可选文件 ===")
optional_files = [
    "README_logits_only.md",      # 使用说明
    "run_with_precomputed_logits.py",  # 批量运行脚本
]

for f in optional_files:
    print(f"  - {f}")