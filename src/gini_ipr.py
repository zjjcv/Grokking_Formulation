#!/usr/bin/env python3
"""
提取 Transformer 权重的输入嵌入矩阵，在频域计算基尼系数和逆参与率

将训练过程中保存的 checkpoint 文件的 token embedding 提取出来，
展平为一维向量，进行 DFT 转换到频域，然后在频域上计算其基尼系数（Gini Coefficient）
和逆参与率（Inverse Participation Ratio），并保存到 CSV 文件。

基尼系数：衡量分布的不平等程度，范围 [0, 1]，值越大表示越不平等
逆参与率：1 / (Σ(x_i / Σx)^2)，衡量分布的有效维度
频域分析：使用 DFT 幅度进行统计
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
    # 确保 values 是一维数组
    values = np.array(values).flatten()

    # 去除零值以避免除零错误
    values = values[values > 0]

    if len(values) == 0:
        return 0.0

    # 排序
    sorted_values = np.sort(values)
    n = len(sorted_values)

    # 计算累积和
    cumsum = np.cumsum(sorted_values)

    # 基尼系数公式: G = (2 * Σ(i * x_i)) / (n * Σx_i) - (n + 1) / n
    # 其中 i 是排序后的索引（从1开始）
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
        ipr: 逆参与率，范围 [1, len(values)]
    """
    # 确保 values 是一维数组
    values = np.array(values).flatten()

    # 取绝对值
    values = np.abs(values)

    # 归一化为概率分布
    total = np.sum(values)

    if total == 0:
        return len(values)  # 如果全为零，返回最大可能的IPR

    probs = values / total

    # 计算逆参与率
    ipr = 1.0 / np.sum(probs ** 2)

    return ipr


def extract_metrics_from_checkpoints(checkpoint_dir, output_file):
    """
    从所有 checkpoint 文件中提取嵌入矩阵，进行 DFT 转换后计算基尼系数和逆参与率

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

            # 提取嵌入矩阵
            embedding_matrix = checkpoint['model_state_dict']['embedding.weight'].numpy()
            # shape: (vocab_size, embed_dim) = (p+1, 128)

            # 展平为一维向量
            flat_embedding = embedding_matrix.flatten()

            # === 新增：DFT 转换到频域 ===
            # 对展平的嵌入向量进行一维 DFT
            dft_embedding = np.fft.fft(flat_embedding)
            # 取 DFT 的幅度（频域能量分布）
            dft_magnitude = np.abs(dft_embedding)

            # 在频域上计算基尼系数
            gini = compute_gini_coefficient(dft_magnitude)

            # 在频域上计算逆参与率
            ipr = compute_inverse_participation_ratio(dft_magnitude)

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
                print(f"Step {step}: Gini={gini:.4f}, IPR={ipr:.4f}, "
                      f"Embedding shape={embedding_matrix.shape}")

        except Exception as e:
            print(f"处理 {checkpoint_path} 时出错: {e}")

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
        ginis = [r['gini'] for r in results]
        iprs = [r['ipr'] for r in results]
        print("\n基尼系数统计:")
        print(f"  最小值: {np.min(ginis):.4f}")
        print(f"  最大值: {np.max(ginis):.4f}")
        print(f"  平均值: {np.mean(ginis):.4f}")
        print(f"  最终值: {ginis[-1]:.4f}")

        print("\n逆参与率统计:")
        print(f"  最小值: {np.min(iprs):.4f}")
        print(f"  最大值: {np.max(iprs):.4f}")
        print(f"  平均值: {np.mean(iprs):.4f}")
        print(f"  最终值: {iprs[-1]:.4f}")
    else:
        print("没有可保存的数据")


def main():
    """主函数"""
    # 配置参数
    checkpoint_dir = "/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints"
    output_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/gini_ip.csv"

    print("=" * 60)
    print("提取 Transformer 嵌入矩阵在频域的基尼系数和逆参与率")
    print("=" * 60)
    print(f"Checkpoint 目录: {checkpoint_dir}")
    print(f"输出文件: {output_file}")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 提取指标
    extract_metrics_from_checkpoints(checkpoint_dir, output_file)


if __name__ == "__main__":
    main()
