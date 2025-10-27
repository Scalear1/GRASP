"""
简化的参数解析模块
"""

import argparse

def parser_add_main_args(parser):
    """添加主要的命令行参数"""
    
    # 基本参数
    parser.add_argument('--dataset', type=str, default='cora', help='dataset name')
    parser.add_argument('--sub_dataset', type=str, default='', help='sub dataset name')
    parser.add_argument('--method', type=str, default='gcn', help='method name')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--cpu', action='store_true', help='use CPU')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--runs', type=int, default=5, help='number of runs')
    
    # 数据分割参数
    parser.add_argument('--train_prop', type=float, default=0.1, help='training proportion')
    parser.add_argument('--valid_prop', type=float, default=0.1, help='validation proportion')
    
    # OOD检测参数
    parser.add_argument('--ood', type=str, default='MSP', help='OOD detection method')
    parser.add_argument('--rocauc', action='store_true', help='use ROCAUC as eval metric')
    
    # GRASP特定参数
    parser.add_argument('--K', type=int, default=8, help='number of layers for belief propagation')
    parser.add_argument('--alpha', type=float, default=0., help='weight for residual connection in propagation')
    parser.add_argument('--T', type=float, default=1.0, help='temperature for Softmax')
    parser.add_argument('--tau1', type=float, default=5, help='threshold to determine s_id and s_ood')
    parser.add_argument('--tau2', type=float, default=50, help='threshold to select train nodes as G')
    parser.add_argument('--delta', type=float, default=1.001, help='weight for G')
    parser.add_argument('--st', type=str, default='top', choices=['top', 'low', 'random', 'test'], help='selection strategy')
    parser.add_argument('--col', action='store_true', help='use col to count connections')
    parser.add_argument('--adj1', action='store_true')
    parser.add_argument('--test', action='store_true', help='whether to augmentate on test')
    
    # 其他OOD方法参数
    parser.add_argument('--neighbors', type=int, default=10, help='neighbors for KNN')
    parser.add_argument('--noise', type=float, default=0., help='param for baseline ODIN and Mahalanobis')
    
    # Logits相关参数
    parser.add_argument('--logits_dir', type=str, default=None, help='directory containing precomputed logits files')
    parser.add_argument('--logits_file', type=str, default=None, help='specific logits file path')
    parser.add_argument('--force_logits', action='store_true', help='force using logits only, exit if not found')
    
    return parser