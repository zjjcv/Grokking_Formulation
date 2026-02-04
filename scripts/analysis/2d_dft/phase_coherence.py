#!/usr/bin/env python3
"""
统一相位相干性分析 - 支持所有四种运算

使用方法:
    python phase_coherence.py --operation x+y
    python phase_coherence.py --all
"""

import os
import csv
import argparse
import torch
import numpy as np
from scipy.stats import linregress

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


def compute_row_phase_r2(embedding_matrix):
    """对嵌入矩阵的每一行进行 DFT，计算相位与频率索引的 R²"""
    n_tokens, d_model = embedding_matrix.shape
    r2_values = []

    for row in range(n_tokens):
        row_data = embedding_matrix[row, :].astype(np.float64)
        dft_result = np.fft.fft(row_data)
        phases = np.angle(dft_result)
        freq_indices = np.arange(d_model, dtype=float)

        try:
            slope, intercept, r_value, p_value, std_err = linregress(freq_indices, phases)
            r2 = r_value ** 2
            r2_values.append(r2)
        except:
            r2_values.append(0.0)

    return np.mean(r2_values), r2_values


def extract_phase_coherence_from_checkpoints(checkpoint_dir, output_file, p=97):
    """从所有 checkpoint 文件中提取相位相干性指标"""
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
            W_E_p = W_E[:p, :]

            mean_r2, r2_per_row = compute_row_phase_r2(W_E_p)

            result = {
                'step': step,
                'train_loss': float(checkpoint['train_loss']),
                'train_acc': float(checkpoint['train_acc']),
                'test_loss': float(checkpoint['test_loss']),
                'test_acc': float(checkpoint['test_acc']),
                'phase_coherence_r2_mean': float(mean_r2),
                'phase_coherence_r2_std': float(np.std(r2_per_row)),
                'phase_coherence_r2_max': float(np.max(r2_per_row)),
                'phase_coherence_r2_min': float(np.min(r2_per_row)),
            }

            results.append(result)

            if step % 10000 == 0:
                print(f"    Step {step}: Phase Coherence R²={mean_r2:.6f}")

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
    parser = argparse.ArgumentParser(description='计算相位相干性')
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
        output_file = f"{OPERATIONS[op_key]['data_dir']}/phase_coherence_2d.csv"

        if not os.path.exists(checkpoint_dir):
            print(f"  跳过: 检查点目录不存在 - {checkpoint_dir}")
            continue

        extract_phase_coherence_from_checkpoints(checkpoint_dir, output_file, args.p)


if __name__ == "__main__":
    main()
