#!/usr/bin/env python3
"""
提取 Transformer 权重的输入嵌入矩阵，计算功率谱密度

将训练过程中保存的 checkpoint 文件的 token embedding 提取出来，
展平为一维向量，计算其功率谱密度（PSD），并保存到 CSV 文件。
"""

import os
import csv
import torch
import numpy as np
from scipy import signal


def compute_psd(embedding_matrix):
    """
    计算嵌入矩阵的功率谱密度

    Args:
        embedding_matrix: (vocab_size, embed_dim) 的嵌入矩阵

    Returns:
        psd: 归一化的功率谱密度，长度为 embed_dim // 2 + 1
    """
    # 将嵌入矩阵展平为一维向量
    flat = embedding_matrix.flatten().astype(np.float32)

    # 去除均值
    flat = flat - np.mean(flat)

    # 计算功率谱密度 (使用 Welch 方法)
    # nperseg 设置为向量长度，确保频率分辨率
    frequencies, psd = signal.welch(flat, nperseg=min(len(flat), 256))

    # 归一化 PSD
    psd = psd / (np.sum(psd) + 1e-10)

    return frequencies, psd


def extract_psd_from_checkpoints(checkpoint_dir, p, output_file):
    """
    从所有 checkpoint 文件中提取嵌入矩阵并计算 PSD

    Args:
        checkpoint_dir: checkpoint 文件目录
        p: 模数（用于确定频率范围 0 到 p/2）
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

            # 计算 PSD
            frequencies, psd = compute_psd(embedding_matrix)

            # 我们只需要频率在 0 到 p/2 范围内的 PSD
            # 由于我们关心的是与模数 p 相关的频率
            max_freq_idx = min(len(frequencies), p // 2 + 1)

            result = {
                'step': step,
                'train_loss': checkpoint['train_loss'],
                'train_acc': checkpoint['train_acc'],
                'test_loss': checkpoint['test_loss'],
                'test_acc': checkpoint['test_acc'],
            }

            # 添加每个频率的 PSD 值
            for i in range(max_freq_idx):
                result[f'freq_{i}'] = float(psd[i])

            results.append(result)

            if step % 100 == 0:
                print(f"Step {step}: Embedding shape={embedding_matrix.shape}, "
                      f"PSD length={len(psd)}, max_freq_idx={max_freq_idx}")

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

        print(f"\nPSD 数据已保存至: {output_file}")
        print(f"共保存 {len(results)} 个时间步的数据")
        print(f"频率范围: 0 到 {p//2}")
    else:
        print("没有可保存的数据")


def main():
    """主函数"""
    # 配置参数
    checkpoint_dir = "/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints"
    output_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/psd.csv"
    p = 97  # 模数

    print("=" * 60)
    print("提取 Transformer 嵌入矩阵的功率谱密度")
    print("=" * 60)
    print(f"Checkpoint 目录: {checkpoint_dir}")
    print(f"输出文件: {output_file}")
    print(f"模数 p: {p}")
    print(f"频率范围: 0 到 {p//2}")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 提取 PSD
    extract_psd_from_checkpoints(checkpoint_dir, p, output_file)


if __name__ == "__main__":
    main()
