#!/usr/bin/env python3
"""
提取 Transformer 权重，计算 2D 相位相干性（行方向分析）

对 W_E 的每一行（token 维度）进行 1D DFT 得到相位谱 φ_k，
计算相位 φ_k 与频率索引 k 的线性回归决定系数 R²。
这是 2D 框架下的相位分析：分析 token 序列方向的相位线性度。

注意：对于相位-频率线性关系分析，行方向的 1D DFT 是合适的方法。
"""

import os
import csv
import torch
import numpy as np
from scipy.stats import linregress


def compute_row_phase_r2(embedding_matrix):
    """
    对嵌入矩阵的每一行进行 DFT，计算相位与频率索引的 R²

    Args:
        embedding_matrix: (p, d_model) 嵌入矩阵

    Returns:
        mean_r2: 所有行 R² 的平均值
        r2_per_row: 每行的 R² 值
    """
    n_tokens, d_model = embedding_matrix.shape
    r2_values = []

    for row in range(n_tokens):
        # 对每一行进行 1D DFT
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


def extract_2d_phase_coherence_from_checkpoints(checkpoint_dir, output_file):
    """从所有 checkpoint 文件中提取 2D 相位相干性指标"""
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("checkpoint_step_") and f.endswith(".pt"):
            step = int(f.replace("checkpoint_step_", "").replace(".pt", ""))
            checkpoint_files.append((step, os.path.join(checkpoint_dir, f)))

    checkpoint_files.sort(key=lambda x: x[0])

    print(f"找到 {len(checkpoint_files)} 个 checkpoint 文件")

    results = []

    for step, checkpoint_path in checkpoint_files:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            W_E = checkpoint['model_state_dict']['embedding.weight'].numpy()

            # 只取前 p 行
            p = 97
            W_E_p = W_E[:p, :]

            # 计算行方向的相位相干性
            mean_r2, r2_per_row = compute_row_phase_r2(W_E_p)

            result = {
                'step': step,
                'train_loss': checkpoint['train_loss'],
                'train_acc': checkpoint['train_acc'],
                'test_loss': checkpoint['test_loss'],
                'test_acc': checkpoint['test_acc'],
                'phase_coherence_r2_mean': float(mean_r2),
                'phase_coherence_r2_std': float(np.std(r2_per_row)),
                'phase_coherence_r2_max': float(np.max(r2_per_row)),
                'phase_coherence_r2_min': float(np.min(r2_per_row)),
            }

            results.append(result)

            if step % 100 == 0:
                print(f"Step {step}: Phase Coherence R²={mean_r2:.6f}")

        except Exception as e:
            print(f"处理 {checkpoint_path} 时出错: {e}")

    if results:
        fieldnames = list(results[0].keys())
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n数据已保存至: {output_file}")
        print(f"共保存 {len(results)} 个时间步的数据")
    else:
        print("没有可保存的数据")


def main():
    checkpoint_dir = "/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints"
    output_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/phase_coherence_2d.csv"

    print("=" * 60)
    print("计算 2D 框架下的相位相干性（行方向分析）")
    print("=" * 60)
    print(f"Checkpoint 目录: {checkpoint_dir}")
    print(f"输出文件: {output_file}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    extract_2d_phase_coherence_from_checkpoints(checkpoint_dir, output_file)


if __name__ == "__main__":
    main()
