#!/usr/bin/env python3
"""
统一傅里叶投影分析 - 支持所有四种运算

计算权重矩阵在空间基和傅里叶基下的投影稀疏度

使用方法:
    python fourier_projection.py --operation x+y
    python fourier_projection.py --all
"""

import os
import csv
import argparse
import torch
import numpy as np

# 添加父目录到路径
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def compute_l1_l2_ratio(matrix, axis=None):
    """计算 L1/L2 比值作为稀疏度指标"""
    matrix = np.abs(matrix)
    l1 = np.sum(matrix, axis=axis)
    l2 = np.sqrt(np.sum(matrix ** 2, axis=axis))
    return l1 / (l2 + 1e-10)


def compute_gini_coefficient(values):
    """计算基尼系数"""
    values = np.array(values).flatten()
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    indices = np.arange(1, n + 1)
    gini = (2 * np.sum(indices * sorted_values) / (n * cumsum[-1]) - (n + 1) / n)
    return gini


def compute_projection_sparsity_2d(weight_matrix):
    """计算权重矩阵在空间基和傅里叶基下的 2D 投影稀疏度"""
    # 空间基（原始矩阵）
    spatial_coeffs = weight_matrix

    # 傅里叶基
    fourier_coeffs = np.fft.fft2(weight_matrix)

    # L1/L2 稀疏度
    spatial_l1l2 = compute_l1_l2_ratio(spatial_coeffs)
    fourier_l1l2 = compute_l1_l2_ratio(fourier_coeffs)

    # Gini 系数
    spatial_gini = compute_gini_coefficient(np.abs(spatial_coeffs))
    fourier_gini = compute_gini_coefficient(np.abs(fourier_coeffs))

    return (spatial_l1l2, fourier_l1l2, spatial_gini, fourier_gini)


def extract_fourier_projection_from_checkpoints(checkpoint_dir, output_file, p=97):
    """从所有 checkpoint 文件中提取 2D 投影稀疏度"""
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("checkpoint_step_") and f.endswith(".pt"):
            step = int(f.replace("checkpoint_step_", "").replace(".pt", ""))
            checkpoint_files.append((step, os.path.join(checkpoint_dir, f)))

    checkpoint_files.sort(key=lambda x: x[0])
    print(f"  找到 {len(checkpoint_files)} 个 checkpoint 文件")

    results = []

    for step, checkpoint_path in checkpoint_files:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            W_E = checkpoint['model_state_dict']['embedding.weight'].numpy()
            W_U = checkpoint['model_state_dict']['output.weight'].numpy()

            W_E = W_E[:p, :]

            (spatial_l1l2_E, fourier_l1l2_E, spatial_gini_E, fourier_gini_E) = \
                compute_projection_sparsity_2d(W_E)
            (spatial_l1l2_U, fourier_l1l2_U, spatial_gini_U, fourier_gini_U) = \
                compute_projection_sparsity_2d(W_U)

            result = {
                'step': step,
                'train_loss': float(checkpoint['train_loss']),
                'train_acc': float(checkpoint['train_acc']),
                'test_loss': float(checkpoint['test_loss']),
                'test_acc': float(checkpoint['test_acc']),
                'W_E_spatial_l1l2': float(spatial_l1l2_E),
                'W_E_spatial_gini': float(spatial_gini_E),
                'W_E_fourier_l1l2': float(fourier_l1l2_E),
                'W_E_fourier_gini': float(fourier_gini_E),
                'W_U_spatial_l1l2': float(spatial_l1l2_U),
                'W_U_spatial_gini': float(spatial_gini_U),
                'W_U_fourier_l1l2': float(fourier_l1l2_U),
                'W_U_fourier_gini': float(fourier_gini_U),
            }

            results.append(result)

            if step % 10000 == 0:
                print(f"    Step {step}: W_E_Spatial={spatial_l1l2_E:.2f}, W_E_Fourier={fourier_l1l2_E:.2f}")

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
    parser = argparse.ArgumentParser(description='计算傅里叶投影稀疏度')
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
        output_file = f"{OPERATIONS[op_key]['data_dir']}/fourier_projection_2d.csv"

        if not os.path.exists(checkpoint_dir):
            print(f"  跳过: 检查点目录不存在 - {checkpoint_dir}")
            continue

        extract_fourier_projection_from_checkpoints(checkpoint_dir, output_file, args.p)


if __name__ == "__main__":
    main()
