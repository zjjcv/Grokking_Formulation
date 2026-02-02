#!/usr/bin/env python3
"""
提取 Transformer 权重的输入嵌入矩阵，计算 2D 功率谱密度

将训练过程中保存的 checkpoint 文件的 token embedding 提取出来，
进行 2D 离散傅里叶变换（2D DFT），计算功率谱密度矩阵，
并保存到 CSV 文件。

2D PSD 分析：
- 对嵌入矩阵 (vocab_size, embed_dim) 进行 2D DFT
- 得到二维频谱矩阵
- 提取关键统计量：DC 分量、低频能量、频谱熵等
"""

import os
import csv
import torch
import numpy as np


def compute_2d_psd(embedding_matrix):
    """
    计算嵌入矩阵的 2D 功率谱密度

    Args:
        embedding_matrix: (vocab_size, embed_dim) 的嵌入矩阵

    Returns:
        psd_2d: 2D 功率谱密度矩阵 (vocab_size, embed_dim)
        psd_magnitude: PSD 幅度谱
        dc_component: DC 分量 (0, 0)
        low_freq_energy: 低频区域能量
        high_freq_energy: 高频区域能量
        spectral_entropy: 频谱熵
    """
    # 进行 2D DFT
    dft_result = np.fft.fft2(embedding_matrix)

    # 计算功率谱密度（幅度平方）
    psd_2d = np.abs(dft_result) ** 2

    # 幅度谱
    psd_magnitude = np.abs(dft_result)

    # DC 分量 (0, 0)
    dc_component = psd_magnitude[0, 0]

    # 低频区域能量（左上角 5x5 区域）
    h, w = psd_2d.shape
    low_freq_size = min(5, h // 2, w // 2)
    low_freq_energy = np.sum(psd_magnitude[:low_freq_size, :low_freq_size])

    # 高频区域能量（右下角 5x5 区域）
    high_freq_energy = np.sum(psd_magnitude[-low_freq_size:, -low_freq_size:])

    # 频谱熵（归一化后的熵）
    psd_normalized = psd_2d / (np.sum(psd_2d) + 1e-10)
    spectral_entropy = -np.sum(psd_normalized * np.log(psd_normalized + 1e-10))

    return psd_2d, psd_magnitude, dc_component, low_freq_energy, high_freq_energy, spectral_entropy


def extract_2d_psd_from_checkpoints(checkpoint_dir, p, output_file):
    """
    从所有 checkpoint 文件中提取嵌入矩阵并计算 2D PSD

    Args:
        checkpoint_dir: checkpoint 文件目录
        p: 模数
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

            # 只取前 p 行（对应 p 个输出类别，去掉操作符行）
            embedding_matrix = embedding_matrix[:p, :]  # (p, embed_dim)

            # 计算 2D PSD
            psd_2d, psd_magnitude, dc_component, low_freq_energy, high_freq_energy, spectral_entropy = \
                compute_2d_psd(embedding_matrix)

            result = {
                'step': step,
                'train_loss': checkpoint['train_loss'],
                'train_acc': checkpoint['train_acc'],
                'test_loss': checkpoint['test_loss'],
                'test_acc': checkpoint['test_acc'],
                'dc_component': float(dc_component),
                'low_freq_energy': float(low_freq_energy),
                'high_freq_energy': float(high_freq_energy),
                'spectral_entropy': float(spectral_entropy),
                'mean_psd': float(np.mean(psd_magnitude)),
                'std_psd': float(np.std(psd_magnitude)),
                'max_psd': float(np.max(psd_magnitude)),
            }

            # 保存完整的 2D PSD（按行展平）
            for i in range(p):
                for j in range(embedding_matrix.shape[1]):
                    result[f'psd_{i}_{j}'] = float(psd_magnitude[i, j])

            results.append(result)

            if step % 100 == 0:
                print(f"Step {step}: DC={dc_component:.2f}, LowFreq={low_freq_energy:.2e}, "
                      f"Entropy={spectral_entropy:.2f}")

        except Exception as e:
            print(f"处理 {checkpoint_path} 时出错: {e}")

    # 保存到 CSV
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
    """主函数"""
    checkpoint_dir = "/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints"
    output_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/psd_2d.csv"
    p = 97  # 模数

    print("=" * 60)
    print("提取 Transformer 嵌入矩阵的 2D 功率谱密度")
    print("=" * 60)
    print(f"Checkpoint 目录: {checkpoint_dir}")
    print(f"输出文件: {output_file}")
    print(f"模数 p: {p}")
    print("=" * 60)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    extract_2d_psd_from_checkpoints(checkpoint_dir, p, output_file)


if __name__ == "__main__":
    main()
