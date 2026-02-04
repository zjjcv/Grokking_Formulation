#!/usr/bin/env python3
"""
有效秩 (Effective Rank) 分析 - 优化版本

对每层线性映射做 SVD，计算谱熵和有效秩: erank(W) = exp(H(W))

使用方法:
    python effective_rank.py --operation x+y
    python effective_rank.py --all
"""

import os
import csv
import argparse
import torch
import numpy as np

# 添加父目录到路径
sys_path = os.path.dirname(os.path.abspath(__file__))
if sys_path not in __file__:
    import sys
    sys.path.insert(0, sys_path)

try:
    from lib.config import OPERATIONS, get_checkpoint_dir
except ImportError:
    OPERATIONS = {
        'x+y': {'name': 'Addition', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x+y'},
        'x-y': {'name': 'Subtraction', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x-y'},
        'x*y': {'name': 'Multiplication', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x*y'},
        'x_div_y': {'name': 'Division', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x_div_y'},
    }
    def get_checkpoint_dir(op): return f"{OPERATIONS[op]['data_dir']}/checkpoints"


def compute_effective_rank(matrix, eps=1e-10):
    """
    计算矩阵的有效秩（优化版：只计算奇异值）

    Args:
        matrix: 输入矩阵 (m, n)
        eps: 小常数避免log(0)

    Returns:
        singular_values: 奇异值
        spectral_entropy: 谱熵 H(W)
        effective_rank: 有效秩 erank(W) = exp(H(W))
    """
    # SVD 分解（只计算奇异值，不计算 U 和 V 矩阵，大幅提高性能）
    s = np.linalg.svd(matrix, compute_uv=False, full_matrices=False)

    # 归一化奇异值
    s_normalized = s / (np.sum(s) + eps)

    # 计算谱熵
    spectral_entropy = -np.sum(s_normalized * np.log(s_normalized + eps))

    # 有效秩 = exp(谱熵)
    effective_rank = np.exp(spectral_entropy)

    return s, spectral_entropy, effective_rank


def analyze_layer_matrices(checkpoint, p, layer_idx=0):
    """
    分析某一层的所有线性矩阵的有效秩

    返回该层的所有有效秩数据
    """
    state_dict = checkpoint['model_state_dict']
    results = {}

    # 1. 输入嵌入 W_E: (p+1, embed_dim) -> 只取前 p 行
    W_E = state_dict['embedding.weight'].numpy()[:p, :]
    _, entropy, erank = compute_effective_rank(W_E)
    results['W_E_entropy'] = float(entropy)
    results['W_E_erank'] = float(erank)

    # 2. 层权重矩阵
    # W_Q: (embed_dim, embed_dim)
    W_Q = state_dict[f'blocks.{layer_idx}.attention.W_q.weight'].numpy()
    _, entropy, erank = compute_effective_rank(W_Q)
    results[f'W_Q_entropy'] = float(entropy)
    results[f'W_Q_erank'] = float(erank)

    # W_K: (embed_dim, embed_dim)
    W_K = state_dict[f'blocks.{layer_idx}.attention.W_k.weight'].numpy()
    _, entropy, erank = compute_effective_rank(W_K)
    results[f'W_K_entropy'] = float(entropy)
    results[f'W_K_erank'] = float(erank)

    # W_V: (embed_dim, embed_dim)
    W_V = state_dict[f'blocks.{layer_idx}.attention.W_v.weight'].numpy()
    _, entropy, erank = compute_effective_rank(W_V)
    results[f'W_V_entropy'] = float(entropy)
    results[f'W_V_erank'] = float(erank)

    # W_O (output projection): (embed_dim, embed_dim)
    W_O = state_dict[f'blocks.{layer_idx}.attention.W_o.weight'].numpy()
    _, entropy, erank = compute_effective_rank(W_O)
    results[f'W_O_entropy'] = float(entropy)
    results[f'W_O_erank'] = float(erank)

    # W_1 (FFN first layer): (embed_dim, 4*embed_dim)
    W_1 = state_dict[f'blocks.{layer_idx}.ffn.linear1.weight'].numpy()
    _, entropy, erank = compute_effective_rank(W_1)
    results[f'W_1_entropy'] = float(entropy)
    results[f'W_1_erank'] = float(erank)

    # W_2 (FFN second layer): (4*embed_dim, embed_dim)
    W_2 = state_dict[f'blocks.{layer_idx}.ffn.linear2.weight'].numpy()
    _, entropy, erank = compute_effective_rank(W_2)
    results[f'W_2_entropy'] = float(entropy)
    results[f'W_2_erank'] = float(erank)

    # 第二层
    if f'blocks.1.attention.W_q.weight' in state_dict:
        layer_idx = 1

        W_Q = state_dict[f'blocks.{layer_idx}.attention.W_q.weight'].numpy()
        _, entropy, erank = compute_effective_rank(W_Q)
        results[f'W_Q_L2_entropy'] = float(entropy)
        results[f'W_Q_L2_erank'] = float(erank)

        W_K = state_dict[f'blocks.{layer_idx}.attention.W_k.weight'].numpy()
        _, entropy, erank = compute_effective_rank(W_K)
        results[f'W_K_L2_entropy'] = float(entropy)
        results[f'W_K_L2_erank'] = float(erank)

        W_V = state_dict[f'blocks.{layer_idx}.attention.W_v.weight'].numpy()
        _, entropy, erank = compute_effective_rank(W_V)
        results[f'W_V_L2_entropy'] = float(entropy)
        results[f'W_V_L2_erank'] = float(erank)

        W_O = state_dict[f'blocks.{layer_idx}.attention.W_o.weight'].numpy()
        _, entropy, erank = compute_effective_rank(W_O)
        results[f'W_O_L2_entropy'] = float(entropy)
        results[f'W_O_L2_erank'] = float(erank)

        W_1 = state_dict[f'blocks.{layer_idx}.ffn.linear1.weight'].numpy()
        _, entropy, erank = compute_effective_rank(W_1)
        results[f'W_1_L2_entropy'] = float(entropy)
        results[f'W_1_L2_erank'] = float(erank)

        W_2 = state_dict[f'blocks.{layer_idx}.ffn.linear2.weight'].numpy()
        _, entropy, erank = compute_effective_rank(W_2)
        results[f'W_2_L2_entropy'] = float(entropy)
        results[f'W_2_L2_erank'] = float(erank)

    return results


def extract_effective_rank_from_checkpoints(checkpoint_dir, output_file, p=97):
    """从所有 checkpoint 文件中提取有效秩数据"""
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("checkpoint_step_") and f.endswith(".pt"):
            step = int(f.replace("checkpoint_step_", "").replace(".pt", ""))
            checkpoint_files.append((step, os.path.join(checkpoint_dir, f)))

    checkpoint_files.sort(key=lambda x: x[0])

    print(f"  找到 {len(checkpoint_files)} 个 checkpoint 文件")

    results = []

    for i, (step, checkpoint_path) in enumerate(checkpoint_files):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 计算有效秩
            erank_data = analyze_layer_matrices(checkpoint, p)

            result = {
                'step': step,
                'train_loss': float(checkpoint['train_loss']),
                'train_acc': float(checkpoint['train_acc']),
                'test_loss': float(checkpoint['test_loss']),
                'test_acc': float(checkpoint['test_acc']),
                **erank_data
            }

            results.append(result)

            # 更频繁的进度输出
            if (i + 1) % 100 == 0 or step == 0:
                print(f"    进度: {i+1}/{len(checkpoint_files)} ({100*(i+1)/len(checkpoint_files):.1f}%) - Step {step}: W_E_erank={result['W_E_erank']:.2f}")

        except Exception as e:
            print(f"    处理 {checkpoint_path} 时出错: {e}")

    if results:
        fieldnames = list(results[0].keys())
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"  数据已保存至: {output_file}")
        print(f"  共保存 {len(results)} 个时间步的数据")
        return True
    else:
        print("  没有可保存的数据")
        return False


def main():
    parser = argparse.ArgumentParser(description='计算有效秩分析')
    parser.add_argument('--operation', type=str, default='x+y',
                        choices=['x+y', 'x-y', 'x*y', 'x_div_y', 'all'],
                        help='模运算类型')
    parser.add_argument('--p', type=int, default=97, help='模数')

    args = parser.parse_args()

    operations = [args.operation] if args.operation != 'all' else list(OPERATIONS.keys())

    for op_key in operations:
        print(f"\n{'='*60}")
        print(f"分析: {OPERATIONS[op_key]['name']} ({op_key})")
        print(f"{'='*60}")

        checkpoint_dir = get_checkpoint_dir(op_key)
        output_file = f"{OPERATIONS[op_key]['data_dir']}/effective_rank.csv"

        if not os.path.exists(checkpoint_dir):
            print(f"  跳过: 检查点目录不存在 - {checkpoint_dir}")
            continue

        extract_effective_rank_from_checkpoints(checkpoint_dir, output_file, args.p)


if __name__ == "__main__":
    main()
