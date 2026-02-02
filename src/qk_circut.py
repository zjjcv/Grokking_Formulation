#!/usr/bin/env python3
"""
提取 Transformer 权重，计算 QK 电路的二维频域分析

计算注意力交互矩阵 A = W_E @ W_Q @ W_K^T @ W_E^T（维度 p x p），
然后进行二维离散傅里叶变换 (2D DFT)，分析频域特征。

QK 电路分析步骤：
1. 提取输入嵌入 W_E: (p+1, embed_dim)
2. 提取查询权重 W_Q: (embed_dim, embed_dim)
3. 提取键权重 W_K: (embed_dim, embed_dim)
4. 计算 A = W_E @ W_Q @ W_K^T @ W_E^T，取前 p 行和列 (p x p)
5. 对 A 进行二维 DFT
6. 保存频谱数据
"""

import os
import csv
import torch
import numpy as np


def compute_2d_dft(matrix):
    """
    对矩阵进行二维 DFT

    Args:
        matrix: (n, m) 矩阵

    Returns:
        dft_matrix: 2D DFT 结果，复数矩阵
        magnitude: 幅度谱
        phase: 相位谱
    """
    dft_matrix = np.fft.fft2(matrix)
    magnitude = np.abs(dft_matrix)
    phase = np.angle(dft_matrix)
    return dft_matrix, magnitude, phase


def compute_qk_circuit_matrix(checkpoint, p, layer_idx=0):
    """
    计算 QK 电路矩阵 A = W_E @ W_Q @ W_K^T @ W_E^T

    Args:
        checkpoint: checkpoint 字典
        p: 模数
        layer_idx: 使用第几层的注意力权重（默认第0层）

    Returns:
        A: QK 电路矩阵 (p, p)
    """
    state_dict = checkpoint['model_state_dict']

    # 提取输入嵌入 W_E: (p+1, embed_dim) = (98, 128)
    W_E = state_dict['embedding.weight'].numpy()

    # 提取查询权重 W_Q: (embed_dim, embed_dim) = (128, 128)
    W_Q = state_dict[f'blocks.{layer_idx}.attention.W_q.weight'].numpy()

    # 提取键权重 W_K: (embed_dim, embed_dim) = (128, 128)
    W_K = state_dict[f'blocks.{layer_idx}.attention.W_k.weight'].numpy()

    # 计算 A = W_E @ W_Q @ W_K^T @ W_E^T
    # W_E: (98, 128), W_Q: (128, 128), W_K^T: (128, 128), W_E^T: (128, 98)
    # 中间结果: W_E @ W_Q = (98, 128)
    # 然后 @ W_K^T = (98, 128)
    # 最后 @ W_E^T = (98, 98)

    temp = W_E @ W_Q  # (98, 128)
    temp = temp @ W_K.T  # (98, 128)
    A = temp @ W_E.T  # (98, 98)

    # 只取前 p 行和列（对应 p 个输出类别，去掉操作符行）
    A_p = A[:p, :p]  # (97, 97)

    return A_p


def analyze_qk_circuit_at_steps(checkpoint_dir, steps, p, output_file):
    """
    在指定步数计算 QK 电路的二维 DFT

    Args:
        checkpoint_dir: checkpoint 文件目录
        steps: 要分析的步数列表
        p: 模数
        output_file: 输出 CSV 文件路径
    """
    results = []

    for step in steps:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt')

        if not os.path.exists(checkpoint_path):
            print(f'警告: Checkpoint {checkpoint_path} 不存在，跳过')
            continue

        try:
            # 加载 checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 计算 QK 电路矩阵
            A = compute_qk_circuit_matrix(checkpoint, p, layer_idx=0)

            # 进行二维 DFT
            dft_A, magnitude, phase = compute_2d_dft(A)

            # 准备结果数据
            result = {
                'step': step,
                'train_loss': checkpoint['train_loss'],
                'train_acc': checkpoint['train_acc'],
                'test_loss': checkpoint['test_loss'],
                'test_acc': checkpoint['test_acc'],
            }

            # 保存完整的幅度谱（展平）
            # 为了便于分析，我们保存几个关键的统计量
            result['magnitude_mean'] = float(np.mean(magnitude))
            result['magnitude_std'] = float(np.std(magnitude))
            result['magnitude_max'] = float(np.max(magnitude))
            result['magnitude_min'] = float(np.min(magnitude))

            # 保存低频能量（前几个频率分量的能量）
            # DC 分量 (0, 0)
            result['dc_component'] = float(magnitude[0, 0])
            # 低频区域能量 (前 5x5 区域)
            low_freq_energy = np.sum(magnitude[:5, :5])
            result['low_freq_energy'] = float(low_freq_energy)
            # 高频区域能量 (最后 5x5 区域)
            high_freq_energy = np.sum(magnitude[-5:, -5:])
            result['high_freq_energy'] = float(high_freq_energy)

            # 频谱熵（衡量频率分布的均匀程度）
            magnitude_norm = magnitude / (np.sum(magnitude) + 1e-10)
            entropy = -np.sum(magnitude_norm * np.log(magnitude_norm + 1e-10))
            result['spectral_entropy'] = float(entropy)

            # 保存完整的二维频谱（按行展平）
            for i in range(p):
                for j in range(p):
                    result[f'mag_{i}_{j}'] = float(magnitude[i, j])

            results.append(result)

            print(f'Step {step}: Acc={checkpoint["test_acc"]:.4f}, '
                  f'DC={result["dc_component"]:.2f}, '
                  f'LowFreq={result["low_freq_energy"]:.2e}, '
                  f'Entropy={result["spectral_entropy"]:.2f}')

        except Exception as e:
            print(f'处理 {checkpoint_path} 时出错: {e}')
            import traceback
            traceback.print_exc()

    # 保存到 CSV
    if results:
        # 获取所有字段名
        fieldnames = list(results[0].keys())

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f'\n数据已保存至: {output_file}')
        print(f'共保存 {len(results)} 个时间步的数据')

        # 打印统计信息
        print('\n频谱分析统计:')
        for r in results:
            print(f"  Step {r['step']}: DC={r['dc_component']:.2f}, "
                  f"LowFreq={r['low_freq_energy']:.2e}, "
                  f"Entropy={r['spectral_entropy']:.2f}")
    else:
        print('没有可保存的数据')


def main():
    """主函数"""
    # 配置参数
    checkpoint_dir = "/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints"
    output_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/qk_circut.csv"
    p = 97  # 模数

    # 要分析的步数
    steps = [100, 500, 1000, 5000, 10000, 50000, 99000]

    print("=" * 60)
    print("计算 QK 电路的二维频域分析")
    print("=" * 60)
    print(f"Checkpoint 目录: {checkpoint_dir}")
    print(f"输出文件: {output_file}")
    print(f"模数 p: {p}")
    print(f"分析步数: {steps}")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 分析 QK 电路
    analyze_qk_circuit_at_steps(checkpoint_dir, steps, p, output_file)


if __name__ == "__main__":
    main()
