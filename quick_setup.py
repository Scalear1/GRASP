#!/usr/bin/env python3
"""
快速设置和测试脚本
"""

import os
import sys

def main():
    print("🚀 GRASP OOD Detection - 快速设置")
    
    # 检查必需文件
    required_files = [
        "test_ood_minimal.py",
        "dataset_minimal.py", 
        "parse_minimal.py",
        "grasp.py",
        "baselines.py", 
        "data_utils.py",
        "logger.py"
    ]
    
    missing_files = []
    for f in required_files:
        if not os.path.exists(f):
            missing_files.append(f)
    
    if missing_files:
        print("❌ 缺少以下必需文件:")
        for f in missing_files:
            print(f"   {f}")
        print("\n💡 请确保所有必需文件都在当前目录中")
        return False
    
    print("✅ 所有必需文件都存在")
    
    # 创建必要的目录
    os.makedirs('results', exist_ok=True)
    os.makedirs('logits', exist_ok=True)
    
    print("✅ 创建了必要的目录")
    
    # 配置提示
    print("\n📝 配置步骤:")
    print("1. 在 dataset_minimal.py 中修改 load_your_dgl_dataset() 函数:")
    print("   - 设置正确的数据集路径")
    print("   - 确认特征和标签字段名")
    print("   - 修改数据集名称")
    
    print("\n2. 准备您的logits文件:")
    print("   - 格式: torch.Tensor, 形状 [num_nodes, num_classes]")
    print("   - 保存为: logits/your_dataset_logits.pt")
    
    print("\n3. 运行测试:")
    print("   python test_ood_minimal.py --dataset your_dataset_name --ood GRASP --logits_file logits/your_dataset_logits.pt --force_logits")
    
    return True

if __name__ == "__main__":
    if main():
        print("\n🎉 设置完成！您可以开始使用GRASP进行OOD检测了。")
    else:
        sys.exit(1)