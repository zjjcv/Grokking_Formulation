#!/usr/bin/env python3
"""
提取 Transformer 权重的输入嵌入矩阵，计算相位相干性

对 W_E 的每一列（特征维度）进行一维离散傅里叶变换（DFT）得到相位谱 φ_k，
定义指标 R² 为相位 φ_k 与频率索引 k 的线性回归决定系数，
衡量 φ_k ∝ k 的程度（相位线性度）。

相位相干性：
- 高 R² 表示相位与频率线性相关（φ ∝ k）
- 低 R² 表示相位随机分布
- 追踪每一列的 R² 并计算均值
"""

import os
import csv
import torch
import numpy as np
from scipy.stats import linregress


def compute_column_phase_r2(embedding_matrix):
    """
    对嵌入矩阵的每一列进行 DFT，计算相位与频率索引的 R²

    Args:
        embedding_matrix: (p, d_model) 嵌入矩阵，p 个 token，d_model 维嵌入

    Returns:
        mean_r2: 所有列 R² 的平均值
        r2_per_column: 每列的 R² 值
        all_phases: 所有列的相位（用于调试）
    """
    n_tokens, d_model = embedding_matrix.shape

    r2_values = []
    all_phases = []

    for col in range(d_model):
        # 对每一列（embedding 维度）进行 DFT
        column = embedding_matrix[:, col].astype(np.float64)

        # 进行一维 DFT
        dft_result = np.fft.fft(column)

        # 提取相位（弧度），范围 [-π, π]
        phases = np.angle(dft_result)

        # 频率索引（0 到 n_tokens-1）
        freq_indices = np.arange(n_tokens, dtype=float)

        # 计算线性回归：phase ~ freq_index
        # 使用 scipy.stats.linregress
        try:
            slope, intercept, r_value, p_value, std_err = linregress(freq_indices, phases)
            r2 = r_value ** 2
            r2_values.append(r2)
        except:
            # 如果计算失败，使用 R² = 0
            r2_values.append(0.0)

        all_phases.append(phases)

    r2_values = np.array(r2_values)

    return np.mean(r2_values), r2_values, all_phases


def extract_phase_coherence_from_checkpoints(checkpoint_dir, output_file):
    """
    从所有 checkpoint 文件中提取相位相干性指标

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

            # 提取输入嵌入 W_E: (vocab_size, embed_dim) = (p+1, 128)
            W_E = checkpoint['model_state_dict']['embedding.weight'].numpy()

            # 只取前 p 行（对应 p 个输出类别，去掉操作符行）
            p = 97  # 模数
            W_E_p = W_E[:p, :]  # (97, 128)

            # 计算相位相干性
            mean_r2, r2_per_column, all_phases = compute_column_phase_r2(W_E_p)

            # 计算统计量
            r2_std = np.std(r2_per_column)
            r2_max = np.max(r2_per_column)
            r2_min = np.min(r2_per_column)
            r2_median = np.median(r2_per_column)

            result = {
                'step': step,
                'train_loss': checkpoint['train_loss'],
                'train_acc': checkpoint['train_acc'],
                'test_loss': checkpoint['test_loss'],
                'test_acc': checkpoint['test_acc'],
                # 相位相干性
                'phase_coherence_r2_mean': float(mean_r2),
                'phase_coherence_r2_std': float(r2_std),
                'phase_coherence_r2_max': float(r2_max),
                'phase_coherence_r2_min': float(r2_min),
                'phase_coherence_r2_median': float(r2_median),
            }

            results.append(result)

            if step % 100 == 0:
                print(f"Step {step}: Phase Coherence R²={mean_r2:.6f}, "
                      f"Std={r2_std:.6f}, Median={r2_median:.6f}")

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
        mean_r2s = [r['phase_coherence_r2_mean'] for r in results]
        median_r2s = [r['phase_coherence_r2_median'] for r in results]

        print("\n相位相干性统计 (Mean R²):")
        print(f"  最小值: {np.min(mean_r2s):.6f}")
        print(f"  最大值: {np.max(mean_r2s):.6f}")
        print(f"  平均值: {np.mean(mean_r2s):.6f}")
        print(f"  初始值: {mean_r2s[0]:.6f}")
        print(f"  最终值: {mean_r2s[-1]:.6f}")

        print("\n相位相干性统计 (Median R²):")
        print(f"  最小值: {np.min(median_r2s):.6f}")
        print(f"  最大值: {np.max(median_r2s):.6f}")
        print(f"  平均值: {np.mean(median_r2s):.6f}")
        print(f"  初始值: {median_r2s[0]:.6f}")
        print(f"  最终值: {median_r2s[-1]:.6f}")
    else:
        print("没有可保存的数据")


def main():
    """主函数"""
    # 配置参数
    checkpoint_dir = "/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints"
    output_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/phase_coherence.csv"

    print("=" * 60)
    print("计算 Transformer 嵌入的相位相干性")
    print("=" * 60)
    print(f"Checkpoint 目录: {checkpoint_dir}")
    print(f"输出文件: {output_file}")
    print("=" * 60)
    print("\n计算方法:")
    print("  对 W_E 的每一列进行一维 DFT")
    print("  提取相位谱 φ_k")
    print("  计算 φ_k 与频率索引 k 的线性回归 R²")
    print("  R² 衡量相位线性度: φ ∝ k")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 提取指标
    extract_phase_coherence_from_checkpoints(checkpoint_dir, output_file)


if __name__ == "__main__":
    main()
