#!/usr/bin/env python3
"""
提取 Transformer 权重的输出嵌入矩阵，计算频域对齐（Frequency Alignment）

计算输入嵌入 $W_E$ 和输出嵌入 $W_U^T$ 在频域上的余弦相似度。
这衡量了输入和输出表示在频域的对齐程度。

频域对齐步骤：
1. 提取输出权重矩阵 $W_U$ (embed_dim × p)
2. 对 $W_U$ 的每一列（p 个输出类别）进行 DFT
3. 对 $W_E$ 进行 DFT
4. 计算频域上的余弦相似度
"""

import os
import csv
import torch
import numpy as np


def compute_dft(matrix):
    """
    对矩阵的每一列进行一维 DFT

    Args:
        matrix: (n, m) 矩阵

    Returns:
        dft_matrix: DFT 结果，形状为 (n, m)
    """
    # 使用 numpy.fft.fft 进行一维 DFT
    dft_matrix = np.fft.fft(matrix, axis=0)
    return dft_matrix


def compute_cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度

    Args:
        vec1, vec2: 向量

    Returns:
        similarity: 余弦相似度，范围 [-1, 1]
    """
    # 计算余弦相似度
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)
    return similarity


def compute_frequency_alignment(checkpoint_dir, p, output_file):
    """
    从所有 checkpoint 文件中计算频域对齐

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

    # 准备输出数据
    results = []

    # 处理每个 checkpoint
    for step, checkpoint_path in checkpoint_files:
        try:
            # 加载 checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 提取输入嵌入 W_E: (vocab_size, embed_dim) = (p+1, 128)
            W_E = checkpoint['model_state_dict']['embedding.weight'].numpy()

            # 提取输出权重 W_U: (p, embed_dim) = (97, 128)
            # PyTorch Linear 层权重形状是 (output_features, input_features)
            W_U = checkpoint['model_state_dict']['output.weight'].numpy()

            # 对 W_E 进行 DFT（沿嵌入维度）
            W_E_dft = np.fft.fft(W_E, axis=1)  # (p+1, 128)
            # 取幅度
            W_E_dft_mag = np.abs(W_E_dft)

            # 对 W_U 的每一行（p 个输出）进行 DFT（沿嵌入维度）
            W_U_dft = np.fft.fft(W_U, axis=1)  # (97, 128)
            # 取幅度
            W_U_dft_mag = np.abs(W_U_dft)

            # 只取前 p 行（对应 p 个输出类别）
            W_E_dft_p = W_E_dft_mag[:p, :]  # (97, 128)

            # 计算频域余弦相似度（对每个输出类别）
            similarities = []
            for i in range(p):
                # W_E_dft_p[i, :] 是第 i 个输出类别的输入嵌入在频域的表示
                # W_U_dft_mag[i, :] 是第 i 个输出类别的输出权重在频域的表示
                sim = compute_cosine_similarity(W_E_dft_p[i, :], W_U_dft_mag[i, :])
                similarities.append(sim)

            # 计算平均相似度
            mean_similarity = np.mean(similarities)

            result = {
                'step': step,
                'train_loss': checkpoint['train_loss'],
                'train_acc': checkpoint['train_acc'],
                'test_loss': checkpoint['test_loss'],
                'test_acc': checkpoint['test_acc'],
                'mean_freq_alignment': float(mean_similarity)
            }

            # 添加每个输出类别的相似度
            for i in range(p):
                result[f'class_{i}_sim'] = float(similarities[i])

            results.append(result)

            if step % 100 == 0:
                print(f"Step {step}: Mean Freq Alignment={mean_similarity:.4f}, "
                      f"W_E shape={W_E.shape}, W_U shape={W_U.shape}")

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
        alignments = [r['mean_freq_alignment'] for r in results]
        print("\n频域对齐统计:")
        print(f"  最小值: {np.min(alignments):.4f}")
        print(f"  最大值: {np.max(alignments):.4f}")
        print(f"  平均值: {np.mean(alignments):.4f}")
        print(f"  初始值: {alignments[0]:.4f}")
        print(f"  最终值: {alignments[-1]:.4f}")
    else:
        print("没有可保存的数据")


def main():
    """主函数"""
    # 配置参数
    checkpoint_dir = "/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints"
    output_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/f_alginment.csv"
    p = 97  # 模数

    print("=" * 60)
    print("计算输入嵌入和输出嵌入的频域对齐")
    print("=" * 60)
    print(f"Checkpoint 目录: {checkpoint_dir}")
    print(f"输出文件: {output_file}")
    print(f"模数 p: {p}")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 计算频域对齐
    compute_frequency_alignment(checkpoint_dir, p, output_file)


if __name__ == "__main__":
    main()
