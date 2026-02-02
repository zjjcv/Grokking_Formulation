#!/usr/bin/env python3
"""
提取 Transformer 权重的输入嵌入矩阵，在频域计算基尼系数和逆参与率

将训练过程中保存的 checkpoint 文件的 token embedding 提取出来，
进行 2D 离散傅里叶变换（2D DFT）到频域，然后计算其基尼系数（Gini Coefficient）
和逆参与率（Inverse Participation Ratio），并保存到 CSV 文件。

2D DFT 版本：
- 对嵌入矩阵进行 2D DFT
- 在二维频域上计算 Gini 系数和 IPR
- 更准确地捕捉二维频域的稀疏度
"""

import os
import csv
import torch
import numpy as np


def compute_gini_coefficient(values):
    """
    计算基尼系数

    Args:
        values: 数值向量

    Returns:
        gini: 基尼系数，范围 [0, 1]
    """
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


def compute_inverse_participation_ratio(values):
    """
    计算逆参与率 (Inverse Participation Ratio, IPR)

    IPR = 1 / Σ(p_i^2)
    其中 p_i = x_i / Σx_i 是归一化的概率分布

    Args:
        values: 数值向量

    Returns:
        ipr: 逆参与率
    """
    values = np.array(values).flatten()
    values = np.abs(values)

    total = np.sum(values)
    if total == 0:
        return len(values)

    probs = values / total
    ipr = 1.0 / np.sum(probs ** 2)

    return ipr


def extract_2d_metrics_from_checkpoints(checkpoint_dir, output_file):
    """
    从所有 checkpoint 文件中提取嵌入矩阵并计算 2D DFT 后的 Gini 系数和 IPR

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

    results = []

    for step, checkpoint_path in checkpoint_files:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            embedding_matrix = checkpoint['model_state_dict']['embedding.weight'].numpy()

            # 只取前 p 行
            p = 97
            embedding_matrix = embedding_matrix[:p, :]

            # 进行 2D DFT
            dft_result = np.fft.fft2(embedding_matrix)

            # 取幅度（频域能量分布）
            magnitude = np.abs(dft_result)

            # 计算基尼系数
            gini = compute_gini_coefficient(magnitude)

            # 计算逆参与率
            ipr = compute_inverse_participation_ratio(magnitude)

            result = {
                'step': step,
                'train_loss': checkpoint['train_loss'],
                'train_acc': checkpoint['train_acc'],
                'test_loss': checkpoint['test_loss'],
                'test_acc': checkpoint['test_acc'],
                'gini': float(gini),
                'ipr': float(ipr)
            }

            results.append(result)

            if step % 100 == 0:
                print(f"Step {step}: Gini={gini:.4f}, IPR={ipr:.4f}")

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
    output_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/gini_ip_2d.csv"

    print("=" * 60)
    print("提取 Transformer 嵌入矩阵的 2D 频域基尼系数和逆参与率")
    print("=" * 60)
    print(f"Checkpoint 目录: {checkpoint_dir}")
    print(f"输出文件: {output_file}")
    print("=" * 60)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    extract_2d_metrics_from_checkpoints(checkpoint_dir, output_file)


if __name__ == "__main__":
    main()
