#!/usr/bin/env python3
"""
提取 Transformer 权重，分析空间基与傅里叶基的投影稀疏度

定义两个基底：
1. B_spatial: 标准 One-hot 基（空间域/原始表示）
2. B_fourier: 离散傅里叶变换基（频域表示）

计算权重矩阵在这两个基底下的投影系数，并计算稀疏度指标。

稀疏度指标：
- L1/L2 比值：||x||_1 / ||x||_2，值越小越稀疏
- Gini 系数：衡量分布不平等程度，值越大越稀疏

追踪"空间稀疏度"与"频域稀疏度"随训练的变化，
验证 IPR 下降是否对应频域稀疏度上升。
"""

import os
import csv
import torch
import numpy as np


def create_fourier_basis(n):
    """
    创建 DFT 傅里叶基矩阵

    Args:
        n: 矩阵大小

    Returns:
        F: n x n 傅里叶基矩阵（复数）
    """
    j, k = np.meshgrid(np.arange(n), np.arange(n))
    F = np.exp(-2j * np.pi * j * k / n) / np.sqrt(n)
    return F


def compute_l1_l2_ratio(matrix, axis=None):
    """
    计算 L1/L2 比值作为稀疏度指标

    Args:
        matrix: 输入矩阵
        axis: 计算的轴（None 表示整体）

    Returns:
        ratio: L1/L2 比值，越小越稀疏
    """
    matrix = np.abs(matrix)
    l1 = np.sum(matrix, axis=axis)
    l2 = np.sqrt(np.sum(matrix ** 2, axis=axis))
    return l1 / (l2 + 1e-10)


def compute_gini_coefficient(values):
    """
    计算基尼系数

    Args:
        values: 数值向量

    Returns:
        gini: 基尼系数，范围 [0, 1]，值越大越不平等（越稀疏）
    """
    values = np.array(values).flatten()
    values = values[values > 0]  # 去除零值

    if len(values) == 0:
        return 0.0

    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    indices = np.arange(1, n + 1)

    gini = (2 * np.sum(indices * sorted_values) / (n * cumsum[-1]) - (n + 1) / n)
    return gini


def compute_projection_sparsity(weight_matrix):
    """
    计算权重矩阵在空间基和傅里叶基下的投影稀疏度

    Args:
        weight_matrix: (m, n) 权重矩阵

    Returns:
        spatial_sparsity_l1l2: 空间域 L1/L2 稀疏度
        fourier_sparsity_l1l2: 频域 L1/L2 稀疏度
        spatial_gini: 空间域 Gini 系数
        fourier_gini: 频域 Gini 系数
    """
    m, n = weight_matrix.shape

    # 空间基（原始矩阵本身就是空间基下的表示）
    spatial_coeffs = weight_matrix

    # 傅里叶基（进行 DFT）
    fourier_coeffs = np.fft.fft2(weight_matrix)

    # 计算 L1/L2 稀疏度
    spatial_sparsity_l1l2 = compute_l1_l2_ratio(spatial_coeffs)
    fourier_sparsity_l1l2 = compute_l1_l2_ratio(fourier_coeffs)

    # 计算 Gini 系数（使用幅度）
    spatial_gini = compute_gini_coefficient(np.abs(spatial_coeffs))
    fourier_gini = compute_gini_coefficient(np.abs(fourier_coeffs))

    return (spatial_sparsity_l1l2, fourier_sparsity_l1l2,
            spatial_gini, fourier_gini)


def extract_fourier_projection_from_checkpoints(checkpoint_dir, output_file):
    """
    从所有 checkpoint 文件中提取空间基和傅里叶基的投影稀疏度

    Args:
        checkpoint_dir: checkpoint 文件目录
        output_file: 输出 CSV 文件路径
    """
    # 获取所有 checkpoint 文件并按步数排序
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("checkpoint_step_") and f.endswith(".pt"):
            step = int(f.replace("checkpoint_step_", "").replace(".pt", ""))
            checkpoint_files.append((step, os.path.join(checkpoint_dir, f)))

    checkpoint_files.sort(key=lambda x: x[0])

    print(f"找到 {len(checkpoint_files)} 个 checkpoint 文件")

    # 准备输出数据
    results = []

    # 处理每个 checkpoint
    for step, checkpoint_path in checkpoint_files:
        try:
            # 加载 checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 提取输入嵌入 W_E: (vocab_size, embed_dim) = (98, 128)
            W_E = checkpoint['model_state_dict']['embedding.weight'].numpy()

            # 提取输出权重 W_U: (p, embed_dim) = (97, 128)
            W_U = checkpoint['model_state_dict']['output.weight'].numpy()

            # 计算 W_E 的投影稀疏度
            (spatial_l1l2_E, fourier_l1l2_E,
             spatial_gini_E, fourier_gini_E) = compute_projection_sparsity(W_E)

            # 计算 W_U 的投影稀疏度
            (spatial_l1l2_U, fourier_l1l2_U,
             spatial_gini_U, fourier_gini_U) = compute_projection_sparsity(W_U)

            # 计算稀疏度比值（频域/空间）
            l1l2_ratio_E = fourier_l1l2_E / (spatial_l1l2_E + 1e-10)
            l1l2_ratio_U = fourier_l1l2_U / (spatial_l1l2_U + 1e-10)

            gini_ratio_E = fourier_gini_E / (spatial_gini_E + 1e-10)
            gini_ratio_U = fourier_gini_U / (spatial_gini_U + 1e-10)

            result = {
                'step': step,
                'train_loss': checkpoint['train_loss'],
                'train_acc': checkpoint['train_acc'],
                'test_loss': checkpoint['test_loss'],
                'test_acc': checkpoint['test_acc'],
                # W_E 空间域稀疏度
                'W_E_spatial_l1l2': float(spatial_l1l2_E),
                'W_E_spatial_gini': float(spatial_gini_E),
                # W_E 频域稀疏度
                'W_E_fourier_l1l2': float(fourier_l1l2_E),
                'W_E_fourier_gini': float(fourier_gini_E),
                # W_E 比值
                'W_E_l1l2_ratio': float(l1l2_ratio_E),
                'W_E_gini_ratio': float(gini_ratio_E),
                # W_U 空间域稀疏度
                'W_U_spatial_l1l2': float(spatial_l1l2_U),
                'W_U_spatial_gini': float(spatial_gini_U),
                # W_U 频域稀疏度
                'W_U_fourier_l1l2': float(fourier_l1l2_U),
                'W_U_fourier_gini': float(fourier_gini_U),
                # W_U 比值
                'W_U_l1l2_ratio': float(l1l2_ratio_U),
                'W_U_gini_ratio': float(gini_ratio_U),
            }

            results.append(result)

            if step % 100 == 0:
                print(f"Step {step}: W_E_Spatial={spatial_l1l2_E:.2f}, W_E_Fourier={fourier_l1l2_E:.2f}, "
                      f"W_U_Spatial={spatial_l1l2_U:.2f}, W_U_Fourier={fourier_l1l2_U:.2f}")

        except Exception as e:
            print(f"处理 {checkpoint_path} 时出错: {e}")
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

        print(f"\n数据已保存至: {output_file}")
        print(f"共保存 {len(results)} 个时间步的数据")

        # 打印统计信息
        spatial_l1l2_E = [r['W_E_spatial_l1l2'] for r in results]
        fourier_l1l2_E = [r['W_E_fourier_l1l2'] for r in results]
        spatial_l1l2_U = [r['W_U_spatial_l1l2'] for r in results]
        fourier_l1l2_U = [r['W_U_fourier_l1l2'] for r in results]

        print("\nW_E 稀疏度统计 (L1/L2):")
        print(f"  空间域: 初始={spatial_l1l2_E[0]:.2f}, 最终={spatial_l1l2_E[-1]:.2f}")
        print(f"  频域: 初始={fourier_l1l2_E[0]:.2f}, 最终={fourier_l1l2_E[-1]:.2f}")

        print("\nW_U 稀疏度统计 (L1/L2):")
        print(f"  空间域: 初始={spatial_l1l2_U[0]:.2f}, 最终={spatial_l1l2_U[-1]:.2f}")
        print(f"  频域: 初始={fourier_l1l2_U[0]:.2f}, 最终={fourier_l1l2_U[-1]:.2f}")

        # 检查消长关系
        spatial_change_E = spatial_l1l2_E[-1] - spatial_l1l2_E[0]
        fourier_change_E = fourier_l1l2_E[-1] - fourier_l1l2_E[0]
        spatial_change_U = spatial_l1l2_U[-1] - spatial_l1l2_U[0]
        fourier_change_U = fourier_l1l2_U[-1] - fourier_l1l2_U[0]

        print("\n消长关系分析:")
        print(f"  W_E: 空间域变化={spatial_change_E:+.2f}, 频域变化={fourier_change_E:+.2f}")
        print(f"  W_U: 空间域变化={spatial_change_U:+.2f}, 频域变化={fourier_change_U:+.2f}")

        if spatial_change_U > 0 and fourier_change_U < 0:
            print(f"  ✓ W_U 呈现消长关系：空间稀疏度上升，频域稀疏度下降")
        else:
            print(f"  ✗ W_U 未呈现典型消长关系")
    else:
        print("没有可保存的数据")


def main():
    """主函数"""
    # 配置参数
    checkpoint_dir = "/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints"
    output_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/fourier_projection.csv"

    print("=" * 60)
    print("计算空间基与傅里叶基的投影稀疏度")
    print("=" * 60)
    print(f"Checkpoint 目录: {checkpoint_dir}")
    print(f"输出文件: {output_file}")
    print("=" * 60)
    print("\n基底定义:")
    print("  B_spatial: 标准 One-hot 基（空间域）")
    print("  B_fourier: 离散傅里叶变换基（频域）")
    print("\n稀疏度指标:")
    print("  L1/L2 比值: 值越小越稀疏")
    print("  Gini 系数: 值越大越稀疏")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 提取指标
    extract_fourier_projection_from_checkpoints(checkpoint_dir, output_file)


if __name__ == "__main__":
    main()
