#!/usr/bin/env python3
"""
提取 Transformer 权重，计算 2D 空间基与傅里叶基的投影稀疏度

定义两个基底：
1. B_spatial: 标准 One-hot 基（空间域/原始表示）
2. B_fourier: 2D 离散傅里叶变换基（频域表示）

计算权重矩阵在这两个基底下的投影系数，并计算稀疏度指标。
"""

import os
import csv
import torch
import numpy as np


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
    """
    计算权重矩阵在空间基和傅里叶基下的 2D 投影稀疏度

    Args:
        weight_matrix: (m, n) 权重矩阵

    Returns:
        spatial_l1l2, fourier_l1l2: L1/L2 稀疏度
        spatial_gini, fourier_gini: Gini 系数
    """
    m, n = weight_matrix.shape

    # 空间基（原始矩阵本身就是空间基下的表示）
    spatial_coeffs = weight_matrix

    # 傅里叶基（进行 2D DFT）
    fourier_coeffs = np.fft.fft2(weight_matrix)

    # 计算 L1/L2 稀疏度
    spatial_l1l2 = compute_l1_l2_ratio(spatial_coeffs)
    fourier_l1l2 = compute_l1_l2_ratio(fourier_coeffs)

    # 计算 Gini 系数（使用幅度）
    spatial_gini = compute_gini_coefficient(np.abs(spatial_coeffs))
    fourier_gini = compute_gini_coefficient(np.abs(fourier_coeffs))

    return (spatial_l1l2, fourier_l1l2, spatial_gini, fourier_gini)


def extract_2d_fourier_projection_from_checkpoints(checkpoint_dir, output_file):
    """从所有 checkpoint 文件中提取 2D 投影稀疏度"""
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
            W_U = checkpoint['model_state_dict']['output.weight'].numpy()

            # 只取前 p 行
            p = 97
            W_E = W_E[:p, :]
            W_U = W_U  # 已经是 (p, embed_dim)

            # 计算 2D 投影稀疏度
            (spatial_l1l2_E, fourier_l1l2_E, spatial_gini_E, fourier_gini_E) = \
                compute_projection_sparsity_2d(W_E)
            (spatial_l1l2_U, fourier_l1l2_U, spatial_gini_U, fourier_gini_U) = \
                compute_projection_sparsity_2d(W_U)

            result = {
                'step': step,
                'train_loss': checkpoint['train_loss'],
                'train_acc': checkpoint['train_acc'],
                'test_loss': checkpoint['test_loss'],
                'test_acc': checkpoint['test_acc'],
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

            if step % 100 == 0:
                print(f"Step {step}: W_E_Spatial={spatial_l1l2_E:.2f}, W_E_Fourier={fourier_l1l2_E:.2f}, "
                      f"W_U_Spatial={spatial_l1l2_U:.2f}, W_U_Fourier={fourier_l1l2_U:.2f}")

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
    output_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/fourier_projection_2d.csv"

    print("=" * 60)
    print("计算 2D 空间基与傅里叶基的投影稀疏度")
    print("=" * 60)
    print(f"Checkpoint 目录: {checkpoint_dir}")
    print(f"输出文件: {output_file}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    extract_2d_fourier_projection_from_checkpoints(checkpoint_dir, output_file)


if __name__ == "__main__":
    main()
