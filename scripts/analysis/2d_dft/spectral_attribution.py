#!/usr/bin/env python3
"""
统一频谱归因分析 - 支持所有四种运算

计算不同频率分量对 Logit 预测的贡献

使用方法:
    python spectral_attribution.py --operation x+y
    python spectral_attribution.py --all
"""

import os
import csv
import argparse
import torch
import numpy as np
from tqdm import tqdm

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


def compute_spectral_attribution_simple(W_e, p, max_freq=20):
    """简化版频谱归因：直接分析频率域的能量分布"""
    W_e_p = W_e[:p, :]

    # 对每列进行 DFT
    W_e_dft = np.fft.fft(W_e_p, axis=0)

    # 计算每个频率的能量
    power_spectrum = np.abs(W_e_dft) ** 2

    # 对特征维度求平均
    freq_energy = power_spectrum.mean(axis=1)

    attributions = {}
    for k in range(min(max_freq, p)):
        attributions[k] = float(freq_energy[k])

    return attributions


def extract_spectral_attribution_from_checkpoints(checkpoint_dir, output_file, p=97, max_freq=20):
    """从所有 checkpoint 文件中提取频谱归因数据"""
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("checkpoint_step_") and f.endswith(".pt"):
            step = int(f.replace("checkpoint_step_", "").replace(".pt", ""))
            checkpoint_files.append((step, os.path.join(checkpoint_dir, f)))

    checkpoint_files.sort(key=lambda x: x[0])
    print(f"  找到 {len(checkpoint_files)} 个 checkpoint 文件")

    results = []

    for step, checkpoint_path in tqdm(checkpoint_files, desc="  处理 checkpoints"):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            W_e = checkpoint['model_state_dict']['embedding.weight'].numpy()

            attributions = compute_spectral_attribution_simple(W_e, p, max_freq)

            result = {
                'step': step,
                'train_loss': float(checkpoint['train_loss']),
                'train_acc': float(checkpoint['train_acc']),
                'test_loss': float(checkpoint['test_loss']),
                'test_acc': float(checkpoint['test_acc']),
            }

            for k, attr in attributions.items():
                result[f'freq_{k}_attribution'] = attr

            # 计算低频和高频贡献
            low_freq_sum = sum(attributions[k] for k in range(min(6, max_freq)))
            high_freq_sum = sum(attributions[k] for k in range(6, max_freq))

            result['low_freq_attribution'] = low_freq_sum
            result['high_freq_attribution'] = high_freq_sum
            result['low_high_ratio'] = low_freq_sum / (high_freq_sum + 1e-10)

            results.append(result)

            if step % 10000 == 0:
                print(f"    Step {step}: Low={low_freq_sum:.4f}, High={high_freq_sum:.4f}, Ratio={result['low_high_ratio']:.4f}")

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
    parser = argparse.ArgumentParser(description='计算频谱归因')
    parser.add_argument('--operation', type=str, default='x+y',
                        choices=['x+y', 'x-y', 'x*y', 'x_div_y', 'all'],
                        help='模运算类型')
    parser.add_argument('--p', type=int, default=97, help='模数')
    parser.add_argument('--max-freq', type=int, default=20, help='分析的最大频率')

    args = parser.parse_args()

    operations = [args.operation] if args.operation != 'all' else list(OPERATIONS.keys())

    for op_key in operations:
        print(f"\n{'='*60}")
        print(f"分析: {OPERATIONS[op_key]['name']} ({op_key})")
        print(f"{'='*60}")

        checkpoint_dir = get_checkpoint_dir(op_key)
        output_file = f"{OPERATIONS[op_key]['data_dir']}/spectral_attribution.csv"

        if not os.path.exists(checkpoint_dir):
            print(f"  跳过: 检查点目录不存在 - {checkpoint_dir}")
            continue

        extract_spectral_attribution_from_checkpoints(checkpoint_dir, output_file, args.p, args.max_freq)


if __name__ == "__main__":
    main()
