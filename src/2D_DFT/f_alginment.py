#!/usr/bin/env python3
"""
提取 Transformer 权重，计算 2D 频域对齐（Frequency Alignment）

计算输入嵌入 $W_E$ 和输出嵌入 $W_U$ 在 2D 频域上的余弦相似度。

2D 频域对齐步骤：
1. 提取输出权重矩阵 W_U: (p, embed_dim)
2. 提取输入嵌入 W_E: (p+1, embed_dim)，取前 p 行
3. 对 W_E 和 W_U 分别进行 2D DFT
4. 计算 2D 频谱的余弦相似度
"""

import os
import csv
import torch
import numpy as np


def compute_cosine_similarity_2d(mat1, mat2):
    """
    计算两个 2D 矩阵的余弦相似度

    Args:
        mat1, mat2: 2D 矩阵，形状相同

    Returns:
        similarity: 余弦相似度，范围 [-1, 1]
    """
    # 展平
    vec1 = mat1.flatten()
    vec2 = mat2.flatten()

    # 计算余弦相似度
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)
    return similarity


def compute_2d_frequency_alignment(checkpoint_dir, p, output_file):
    """
    从所有 checkpoint 文件中计算 2D 频域对齐

    Args:
        checkpoint_dir: checkpoint 文件目录
        p: 模数（输出类别数）
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

            # 提取输入嵌入 W_E: (p+1, embed_dim)
            W_E = checkpoint['model_state_dict']['embedding.weight'].numpy()

            # 提取输出权重 W_U: (p, embed_dim)
            W_U = checkpoint['model_state_dict']['output.weight'].numpy()

            # 只取前 p 行（对应 p 个输出类别）
            W_E_p = W_E[:p, :]  # (p, embed_dim)

            # 进行 2D DFT
            W_E_dft = np.fft.fft2(W_E_p)
            W_U_dft = np.fft.fft2(W_U)

            # 取幅度
            W_E_dft_mag = np.abs(W_E_dft)
            W_U_dft_mag = np.abs(W_U_dft)

            # 计算 2D 频谱的余弦相似度
            similarity = compute_cosine_similarity_2d(W_E_dft_mag, W_U_dft_mag)

            # 计算其他统计量
            w_e_energy = np.sum(W_E_dft_mag ** 2)
            w_u_energy = np.sum(W_U_dft_mag ** 2)
            dc_E = W_E_dft_mag[0, 0]
            dc_U = W_U_dft_mag[0, 0]

            result = {
                'step': step,
                'train_loss': checkpoint['train_loss'],
                'train_acc': checkpoint['train_acc'],
                'test_loss': checkpoint['test_loss'],
                'test_acc': checkpoint['test_acc'],
                'mean_2d_alignment': float(similarity),
                'w_e_energy': float(w_e_energy),
                'w_u_energy': float(w_u_energy),
                'dc_component_E': float(dc_E),
                'dc_component_U': float(dc_U),
            }

            results.append(result)

            if step % 100 == 0:
                print(f"Step {step}: 2D Alignment={similarity:.4f}, "
                      f"W_E shape={W_E_p.shape}, W_U shape={W_U.shape}")

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

        # 打印统计信息
        alignments = [r['mean_2d_alignment'] for r in results]
        print("\n2D 频域对齐统计:")
        print(f"  最小值: {np.min(alignments):.4f}")
        print(f"  最大值: {np.max(alignments):.4f}")
        print(f"  平均值: {np.mean(alignments):.4f}")
        print(f"  初始值: {alignments[0]:.4f}")
        print(f"  最终值: {alignments[-1]:.4f}")
    else:
        print("没有可保存的数据")


def main():
    checkpoint_dir = "/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints"
    output_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/f_alginment_2d.csv"
    p = 97

    print("=" * 60)
    print("计算 2D 频域对齐（W_E 和 W_U）")
    print("=" * 60)
    print(f"Checkpoint 目录: {checkpoint_dir}")
    print(f"输出文件: {output_file}")
    print(f"模数 p: {p}")
    print("=" * 60)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    compute_2d_frequency_alignment(checkpoint_dir, p, output_file)


if __name__ == "__main__":
    main()
