#!/usr/bin/env python3
"""
统一电路竞争分析 - 支持所有四种运算

计算记忆电路与算法电路的竞争关系

使用方法:
    python circuit_competition.py --operation x+y
    python circuit_competition.py --all
"""

import os
import csv
import argparse
import torch
import numpy as np
from tqdm import tqdm
from scipy.linalg import svd

# 添加父目录到路径
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in __file__:
    import sys
    sys.path.insert(0, sys_path)

try:
    from lib.config import OPERATIONS, get_checkpoint_dir
except ImportError:
    OPERATIONS = {
        'x+y': {'name': 'Addition', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x+y'},
        'x-y': {'name': 'Subtraction', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x-y'},
        'x*y': {'name': 'Multiplication', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x*y'},
        'x_div_y': {'name': 'Division', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x_div_y'},
    }
    def get_checkpoint_dir(op): return f"{OPERATIONS[op]['data_dir']}/checkpoints"


def create_fourier_subspace_dynamic(embedding_matrix, n_freqs=10):
    """
    创建动态傅里叶子空间：从权重的 DFT 中提取低频结构

    方法：
    1. 对嵌入矩阵进行 2D DFT
    2. 识别低频区域（左上角）
    3. 通过逆 DFT 重建对应的原始空间基向量
    4. 对低频成分进行 SVD 提取主要方向

    Args:
        embedding_matrix: (vocab_size, embed_dim) 嵌入矩阵
        n_freqs: 使用的低频分量数量

    Returns:
        fourier_basis: (embed_dim, n_freqs) 算法子空间基矩阵
    """
    # 进行 2D DFT
    dft_result = np.fft.fft2(embedding_matrix)

    # 创建低频模板（只保留低频，其他置零）
    low_freq_mask = np.zeros_like(dft_result)
    vocab_size, embed_dim = embedding_matrix.shape

    # 保留左上角的低频区域
    freq_region = min(n_freqs, vocab_size // 2, embed_dim // 2)
    for i in range(freq_region):
        for j in range(freq_region):
            low_freq_mask[i, j] = 1.0
            # 对称的高频部分（DFT 的性质）
            if i > 0:
                low_freq_mask[-i, j] = 1.0
            if j > 0:
                low_freq_mask[i, -j] = 1.0

    # 应用低频模板
    low_freq_dft = dft_result * low_freq_mask

    # 逆 DFT 得到低频成分
    low_freq_component = np.fft.ifft2(low_freq_dft).real

    # 对低频成分进行 SVD，提取主要方向作为基
    U, S, Vt = svd(low_freq_component, full_matrices=False)

    # 取前 n_freqs 个右奇异向量作为基
    fourier_basis = Vt[:n_freqs, :].T  # (embed_dim, n_freqs)

    return fourier_basis


def create_memo_subspace_simple(embedding_matrix, n_components=10):
    """
    记忆子空间：使用嵌入矩阵的主成分（整体方差方向）

    Args:
        embedding_matrix: (vocab_size, embed_dim) 嵌入矩阵
        n_components: 主成分数量

    Returns:
        memo_basis: (embed_dim, n_components) 记忆子空间基
    """
    # 对嵌入矩阵进行 SVD
    U, S, Vt = svd(embedding_matrix, full_matrices=False)

    # 取前 n_components 个右奇异向量作为记忆子空间基
    memo_basis = Vt[:n_components, :].T  # (embed_dim, n_components)

    return memo_basis


def compute_projection_energy(weight_matrix, subspace_basis):
    """
    计算权重矩阵在子空间上的投影能量

    使用正确的投影公式：
    - 投影矩阵 P = V @ V^T （假设 V 是正交归一的）
    - 投影后的矩阵 = W @ V @ V^T
    - 投影能量 = ||投影后的矩阵||²_F

    Args:
        weight_matrix: (m, n) 权重矩阵
        subspace_basis: (n, k) 子空间基矩阵（k 个基向量，每列是一个基向量）

    Returns:
        projection_energy: 投影能量（标量）
    """
    V = subspace_basis
    V_T = V.T

    # 投影：W @ V @ V^T
    projected = weight_matrix @ V @ V_T

    # 计算投影能量（Frobenius 范数的平方）
    projection_energy = np.sum(projected ** 2)

    return projection_energy


def compute_circuit_competition(W_E, W_U, p, n_memo_components=10, n_fourier_freqs=10):
    """
    计算电路竞争指标

    Args:
        W_E: (vocab_size, embed_dim) 输入嵌入矩阵
        W_U: (p, embed_dim) 输出权重矩阵
        p: 模数
        n_memo_components: 记忆子空间主成分数
        n_fourier_freqs: 傅里叶子空间低频分量数

    Returns:
        dict: 包含各种能量指标
    """
    # 为 W_E 创建子空间基
    fourier_basis_E = create_fourier_subspace_dynamic(W_E, n_freqs=n_fourier_freqs)
    memo_basis_E = create_memo_subspace_simple(W_E, n_components=n_memo_components)

    # 为 W_U 创建子空间基
    fourier_basis_U = create_fourier_subspace_dynamic(W_U, n_freqs=n_fourier_freqs)
    memo_basis_U = create_memo_subspace_simple(W_U, n_components=n_memo_components)

    # 计算 W_E 在各子空间的投影能量
    memo_energy_E = compute_projection_energy(W_E, memo_basis_E)
    fourier_energy_E = compute_projection_energy(W_E, fourier_basis_E)
    total_energy_E = np.sum(W_E ** 2)
    residual_energy_E = total_energy_E - memo_energy_E - fourier_energy_E

    # 计算 W_U 在各子空间的投影能量
    memo_energy_U = compute_projection_energy(W_U, memo_basis_U)
    fourier_energy_U = compute_projection_energy(W_U, fourier_basis_U)
    total_energy_U = np.sum(W_U ** 2)
    residual_energy_U = total_energy_U - memo_energy_U - fourier_energy_U

    # 计算竞争比率（算法/记忆）
    competition_ratio_E = fourier_energy_E / (memo_energy_E + 1e-10)
    competition_ratio_U = fourier_energy_U / (memo_energy_U + 1e-10)

    return {
        'W_E_fourier_energy': float(fourier_energy_E),
        'W_E_memo_energy': float(memo_energy_E),
        'W_E_residual_energy': float(residual_energy_E),
        'W_E_total_energy': float(total_energy_E),
        'W_E_competition_ratio': float(competition_ratio_E),
        'W_U_fourier_energy': float(fourier_energy_U),
        'W_U_memo_energy': float(memo_energy_U),
        'W_U_residual_energy': float(residual_energy_U),
        'W_U_total_energy': float(total_energy_U),
        'W_U_competition_ratio': float(competition_ratio_U),
    }


def extract_circuit_competition_from_checkpoints(checkpoint_dir, output_file, p=97, d_model=128,
                                                 n_memo_components=10, n_fourier_freqs=10):
    """
    从所有 checkpoint 文件中提取电路竞争数据

    Args:
        checkpoint_dir: checkpoint 目录
        output_file: 输出 CSV 文件
        p: 模数
        d_model: 模型维度
        n_memo_components: 记忆子空间主成分数
        n_fourier_freqs: 傅里叶子空间低频分量数
    """
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("checkpoint_step_") and f.endswith(".pt"):
            step = int(f.replace("checkpoint_step_", "").replace(".pt", ""))
            checkpoint_files.append((step, os.path.join(checkpoint_dir, f)))

    checkpoint_files.sort(key=lambda x: x[0])
    print(f"  找到 {len(checkpoint_files)} 个 checkpoint 文件")

    results = []

    for step, checkpoint_path in tqdm(checkpoint_files, desc="  处理 checkpoints"):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            W_E = checkpoint['model_state_dict']['embedding.weight'].numpy()
            W_U = checkpoint['model_state_dict']['output.weight'].numpy()

            # 只取前 p 行（排除特殊 token）
            W_E = W_E[:p, :]

            # 计算电路竞争指标（一次性计算 W_E 和 W_U）
            metrics = compute_circuit_competition(W_E, W_U, p, n_memo_components, n_fourier_freqs)

            result = {
                'step': step,
                'train_loss': float(checkpoint['train_loss']),
                'train_acc': float(checkpoint['train_acc']),
                'test_loss': float(checkpoint['test_loss']),
                'test_acc': float(checkpoint['test_acc']),
                **metrics  # 包含所有 W_E 和 W_U 的指标
            }

            results.append(result)

            if step % 10000 == 0:
                print(f"    Step {step}: W_U Ratio={metrics['W_U_competition_ratio']:.4f}")

        except Exception as e:
            print(f"    处理 {checkpoint_path} 时出错: {e}")

    if results:
        fieldnames = list(results[0].keys())
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"  数据已保存至: {output_file}")
        print(f"  共保存 {len(results)} 个时间步的数据")

        # 寻找 crossover 点
        ratios = [r['W_U_competition_ratio'] for r in results]
        steps = [r['step'] for r in results]
        for i in range(1, len(ratios)):
            if ratios[i-1] < 1.0 and ratios[i] >= 1.0:
                print(f"  Crossover 点: Step {steps[i]}")
                break
        return True
    else:
        print("  没有可保存的数据")
        return False


def main():
    parser = argparse.ArgumentParser(description='计算电路竞争')
    parser.add_argument('--operation', type=str, default='x+y',
                        choices=['x+y', 'x-y', 'x*y', 'x_div_y', 'all'],
                        help='模运算类型')
    parser.add_argument('--p', type=int, default=97, help='模数')
    parser.add_argument('--d-model', type=int, default=128, help='模型维度')
    parser.add_argument('--n-memo-components', type=int, default=10,
                        help='记忆子空间主成分数')
    parser.add_argument('--n-fourier-freqs', type=int, default=10,
                        help='傅里叶子空间低频分量数')

    args = parser.parse_args()

    operations = [args.operation] if args.operation != 'all' else list(OPERATIONS.keys())

    for op_key in operations:
        print(f"\n{'='*60}")
        print(f"分析: {OPERATIONS[op_key]['name']} ({op_key})")
        print(f"{'='*60}")

        checkpoint_dir = get_checkpoint_dir(op_key)
        output_file = f"{OPERATIONS[op_key]['data_dir']}/circuit_Competition.csv"

        if not os.path.exists(checkpoint_dir):
            print(f"  跳过: 检查点目录不存在 - {checkpoint_dir}")
            continue

        extract_circuit_competition_from_checkpoints(
            checkpoint_dir, output_file, args.p, args.d_model,
            args.n_memo_components, args.n_fourier_freqs
        )


if __name__ == "__main__":
    main()
