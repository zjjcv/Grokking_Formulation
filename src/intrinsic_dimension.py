#!/usr/bin/env python3
"""
提取 Transformer 权重，计算内在维度和有效秩

计算以下指标：
1. 使用 TwoNN 方法计算 W_E 和 W_U 的内在维度（Intrinsic Dimension, ID）
2. 对 W_QK = W_Q @ W_K^T 进行 SVD 分解，计算有效秩和谱熵

TwoNN 方法：
- 基于最近邻距离估计内在维度
- ID ≈ -1 / mean(log(μ₂/μ₁))，其中 μ₁, μ₂ 是第一、第二近邻距离

有效秩：
- effective_rank = exp(Σ(p_i * log(p_i)))，p_i = σ_i / Σσ_i
- 衡量矩阵的有效维度

谱熵：
- spectral_entropy = -Σ(p_i * log(p_i))
"""

import os
import csv
import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd


def compute_two_nn_id(embedding, n_samples=1000, k=10):
    """
    使用改进的 TwoNN 方法计算内在维度

    使用 k 个近邻而不是只用 2 个，提高稳定性

    Args:
        embedding: (n, d) 嵌入矩阵
        n_samples: 采样的点数（加速计算）
        k: 使用的近邻数量

    Returns:
        id estimate: 内在维度估计值
    """
    # 如果点太多，进行随机采样
    n = embedding.shape[0]
    if n > n_samples:
        indices = np.random.choice(n, n_samples, replace=False)
        emb_sample = embedding[indices]
    else:
        emb_sample = embedding

    # 计算所有点对之间的距离
    distances = pdist(emb_sample, metric='euclidean')
    dist_matrix = squareform(distances)

    # 对每个点，找到前 k 个近邻的距离
    n_points = emb_sample.shape[0]
    log_ratios_all = []

    for i in range(n_points):
        # 获取第 i 个点到所有其他点的距离，排序
        row_dists = dist_matrix[i]
        sorted_dists = np.sort(row_dists)
        # sorted_dists[0] = 0（到自身的距离）
        # 使用前 k+1 个近邻（排除自身）
        if len(sorted_dists) > k + 1:
            for j in range(1, k + 1):
                for l in range(j + 1, k + 2):
                    if sorted_dists[l] > sorted_dists[j] > 0:
                        ratio = sorted_dists[l] / sorted_dists[j]
                        log_ratio = np.log(ratio)
                        # 只使用正值的 log_ratio（保证 ID 为正）
                        if log_ratio > 0:
                            log_ratios_all.append(log_ratio)

    if len(log_ratios_all) == 0:
        return embedding.shape[1]  # 退回到嵌入维度

    log_ratios_all = np.array(log_ratios_all)

    # 使用中位数而不是均值，更鲁棒
    median_log_ratio = np.median(log_ratios_all)

    # TwoNN 估计: ID ≈ 1 / median(log(ratio))
    if median_log_ratio > 0:
        id_estimate = 1.0 / median_log_ratio
    else:
        # 如果仍然为负或零，使用均值绝对值的倒数
        mean_abs_log_ratio = np.mean(np.abs(log_ratios_all))
        if mean_abs_log_ratio > 0:
            id_estimate = 1.0 / mean_abs_log_ratio
        else:
            id_estimate = embedding.shape[1]  # 退回到嵌入维度

    # 限制 ID 在合理范围内 [1, embedding_dim]
    id_estimate = max(1.0, min(id_estimate, embedding.shape[1]))

    return id_estimate


def compute_effective_rank_and_spectral_entropy(matrix):
    """
    对矩阵进行 SVD 分解，计算有效秩和谱熵

    Args:
        matrix: (m, n) 矩阵

    Returns:
        effective_rank: 有效秩
        spectral_entropy: 谱熵
    """
    # 进行 SVD 分解
    s = svd(matrix, compute_uv=False, full_matrices=False)

    # 归一化奇异值
    s = s[s > 0]  # 只保留非零奇异值
    total = np.sum(s)
    p = s / total

    # 计算谱熵
    spectral_entropy = -np.sum(p * np.log(p + 1e-10))

    # 计算有效秩（指数熵）
    effective_rank = np.exp(spectral_entropy)

    return effective_rank, spectral_entropy


def compute_intrinsic_dimension_metrics(checkpoint_dir, output_file):
    """
    从所有 checkpoint 文件中计算内在维度指标

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

            # 提取输入嵌入 W_E: (p+1, embed_dim) = (98, 128)
            W_E = checkpoint['model_state_dict']['embedding.weight'].numpy()

            # 提取输出权重 W_U: (p, embed_dim) = (97, 128)
            W_U = checkpoint['model_state_dict']['output.weight'].numpy()

            # 提取 W_Q 和 W_K: (embed_dim, embed_dim) = (128, 128)
            W_Q = checkpoint['model_state_dict']['blocks.0.attention.W_q.weight'].numpy()
            W_K = checkpoint['model_state_dict']['blocks.0.attention.W_k.weight'].numpy()

            # 计算 W_QK = W_Q @ W_K^T
            W_QK = W_Q @ W_K.T

            # 计算 W_E 和 W_U 的内在维度（TwoNN 方法）
            id_W_E = compute_two_nn_id(W_E, n_samples=100)
            id_W_U = compute_two_nn_id(W_U, n_samples=100)

            # 计算 W_E 和 W_U 的有效秩和谱熵
            eff_rank_W_E, entropy_W_E = compute_effective_rank_and_spectral_entropy(W_E)
            eff_rank_W_U, entropy_W_U = compute_effective_rank_and_spectral_entropy(W_U)

            # 计算 W_QK 的有效秩和谱熵
            eff_rank_W_QK, entropy_W_QK = compute_effective_rank_and_spectral_entropy(W_QK)

            result = {
                'step': step,
                'train_loss': checkpoint['train_loss'],
                'train_acc': checkpoint['train_acc'],
                'test_loss': checkpoint['test_loss'],
                'test_acc': checkpoint['test_acc'],
                'id_W_E': float(id_W_E),
                'id_W_U': float(id_W_U),
                'eff_rank_W_E': float(eff_rank_W_E),
                'eff_rank_W_U': float(eff_rank_W_U),
                'eff_rank_W_QK': float(eff_rank_W_QK),
                'entropy_W_E': float(entropy_W_E),
                'entropy_W_U': float(entropy_W_U),
                'entropy_W_QK': float(entropy_W_QK),
            }

            results.append(result)

            if step % 100 == 0:
                print(f"Step {step}: ID_W_E={id_W_E:.2f}, ID_W_U={id_W_U:.2f}, "
                      f"EffRank_W_QK={eff_rank_W_QK:.2f}")

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
        id_W_Es = [r['id_W_E'] for r in results]
        id_W_Us = [r['id_W_U'] for r in results]
        eff_rank_W_QKs = [r['eff_rank_W_QK'] for r in results]

        print("\n内在维度统计 (W_E):")
        print(f"  最小值: {np.min(id_W_Es):.2f}")
        print(f"  最大值: {np.max(id_W_Es):.2f}")
        print(f"  平均值: {np.mean(id_W_Es):.2f}")
        print(f"  初始值: {id_W_Es[0]:.2f}")
        print(f"  最终值: {id_W_Es[-1]:.2f}")

        print("\n内在维度统计 (W_U):")
        print(f"  最小值: {np.min(id_W_Us):.2f}")
        print(f"  最大值: {np.max(id_W_Us):.2f}")
        print(f"  平均值: {np.mean(id_W_Us):.2f}")
        print(f"  初始值: {id_W_Us[0]:.2f}")
        print(f"  最终值: {id_W_Us[-1]:.2f}")

        print("\n有效秩统计 (W_QK):")
        print(f"  最小值: {np.min(eff_rank_W_QKs):.2f}")
        print(f"  最大值: {np.max(eff_rank_W_QKs):.2f}")
        print(f"  平均值: {np.mean(eff_rank_W_QKs):.2f}")
        print(f"  初始值: {eff_rank_W_QKs[0]:.2f}")
        print(f"  最终值: {eff_rank_W_QKs[-1]:.2f}")
    else:
        print("没有可保存的数据")


def main():
    """主函数"""
    # 配置参数
    checkpoint_dir = "/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints"
    output_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/intrinsic_dimension.csv"

    print("=" * 60)
    print("计算 Transformer 权重的内在维度和有效秩")
    print("=" * 60)
    print(f"Checkpoint 目录: {checkpoint_dir}")
    print(f"输出文件: {output_file}")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 计算指标
    compute_intrinsic_dimension_metrics(checkpoint_dir, output_file)


if __name__ == "__main__":
    main()
