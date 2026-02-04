#!/usr/bin/env python3
"""
统一 QK 电路 2D 频域分析 - 支持所有四种运算

计算注意力交互矩阵 A = W_E @ W_Q @ W_K^T @ W_E^T 的二维 DFT 分析

使用方法:
    python qk_circut.py --operation x+y
    python qk_circut.py --all
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


def compute_2d_dft(matrix):
    """对矩阵进行二维 DFT"""
    dft_matrix = np.fft.fft2(matrix)
    magnitude = np.abs(dft_matrix)
    phase = np.angle(dft_matrix)
    return dft_matrix, magnitude, phase


def compute_qk_circuit_matrix(checkpoint, p, layer_idx=0):
    """
    计算 QK 电路矩阵 A = W_E @ W_Q @ W_K^T @ W_E^T
    """
    state_dict = checkpoint['model_state_dict']

    # 提取输入嵌入 W_E: (p+1, embed_dim) = (98, 128)
    W_E = state_dict['embedding.weight'].numpy()

    # 提取查询权重 W_Q: (embed_dim, embed_dim) = (128, 128)
    W_Q = state_dict[f'blocks.{layer_idx}.attention.W_q.weight'].numpy()

    # 提取键权重 W_K: (embed_dim, embed_dim) = (128, 128)
    W_K = state_dict[f'blocks.{layer_idx}.attention.W_k.weight'].numpy()

    # 计算 A = W_E @ W_Q @ W_K^T @ W_E^T
    temp = W_E @ W_Q  # (98, 128)
    temp = temp @ W_K.T  # (98, 128)
    A = temp @ W_E.T  # (98, 98)

    # 只取前 p 行和列（对应 p 个输出类别，去掉操作符行）
    A_p = A[:p, :p]  # (97, 97)

    return A_p


def extract_qk_circuit_from_checkpoints(checkpoint_dir, output_file, p=97):
    """
    从所有 checkpoint 文件中提取 QK 电路的 2D DFT 数据
    """
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

            # 计算 QK 电路矩阵
            A = compute_qk_circuit_matrix(checkpoint, p)

            # 进行二维 DFT
            _, magnitude, phase = compute_2d_dft(A)

            result = {
                'step': step,
                'train_loss': float(checkpoint['train_loss']),
                'train_acc': float(checkpoint['train_acc']),
                'test_loss': float(checkpoint['test_loss']),
                'test_acc': float(checkpoint['test_acc']),
                'magnitude_mean': float(np.mean(magnitude)),
                'magnitude_std': float(np.std(magnitude)),
                'magnitude_max': float(np.max(magnitude)),
                'magnitude_min': float(np.min(magnitude)),
                'dc_component': float(magnitude[0, 0]),
                'low_freq_energy': float(np.sum(magnitude[:5, :5])),
                'high_freq_energy': float(np.sum(magnitude[-5:, -5:])),
                'spectral_entropy': float(-np.sum(
                    (magnitude / (np.sum(magnitude) + 1e-10)) *
                    np.log(magnitude / (np.sum(magnitude) + 1e-10) + 1e-10)
                )),
            }

            results.append(result)

            if step % 10000 == 0:
                print(f"    Step {step}: DC={result['dc_component']:.2f}, "
                      f"LowFreq={result['low_freq_energy']:.2e}, "
                      f"Entropy={result['spectral_entropy']:.2f}")

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
    parser = argparse.ArgumentParser(description='计算 QK 电路的 2D 频域分析')
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
        output_file = f"{OPERATIONS[op_key]['data_dir']}/qk_circut.csv"

        if not os.path.exists(checkpoint_dir):
            print(f"  跳过: 检查点目录不存在 - {checkpoint_dir}")
            continue

        extract_qk_circuit_from_checkpoints(checkpoint_dir, output_file, args.p)


if __name__ == "__main__":
    main()
