#!/usr/bin/env python3
"""
群论表示分析 (Group Representation Analysis) - 优化版本

分析 token 嵌入是否形成群结构：
- 对输入嵌入 W_E 中每个数对应的表示列 E_x
- 计算最优旋转矩阵 R_t 最小化 Σ||E_{x+1} - R_t E_x||²_F
- 计算相对残差 ε_R(t) = Σ||E_{x+1} - R_t E_x||²_F / Σ||E_{x+1}||²_F
- 计算正交性 δ_orth(t) = ||R_t^T R_t - I||_F

使用方法:
    python group_representation.py --operation x+y
    python group_representation.py --all
"""

import os
import csv
import argparse
import torch
import numpy as np
from tqdm import tqdm


def compute_optimal_rotation_vectorized(E_current, E_next):
    """
    向量化计算最优旋转矩阵 R (最小化 ||E_next - R * E_current||_F^2)

    使用 SVD 方法: R = V * U^T，其中 E_current * E_next^T = U * Σ * V^T

    Args:
        E_current: (p, embed_dim) 当前嵌入矩阵
        E_next: (p, embed_dim) 下一个嵌入矩阵

    Returns:
        R: (embed_dim, embed_dim) 最优旋转矩阵
    """
    # 计算 M = E_current^T @ E_next
    M = E_current.T @ E_next  # (embed_dim, embed_dim)

    # SVD 分解
    U, _, Vt = np.linalg.svd(M)

    # 最优旋转矩阵: R = V @ U^T
    R = Vt.T @ U.T

    # 确保行列式为 1（而非 -1，即反射）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def compute_group_representation_metrics(W_E, p=97):
    """
    计算群表示指标（优化版本）

    根据公式：
    - R_t = arg_R_min Σ_x ||E_{x+1} - R * E_x||²_F
    - ε_R(t) = Σ_x ||E_{x+1} - R_t * E_x||²_F / Σ_x ||E_{x+1}||²_F
    - δ_orth(t) = ||R_t^T * R_t - I||_F

    Args:
        W_E: (vocab_size, embed_dim) 嵌入矩阵
        p: 模数

    Returns:
        dict: 包含相对残差和正交性系数
    """
    # 只取前 p 个 token (0 到 p-1)
    E = W_E[:p, :]  # (p, embed_dim)

    # 构造连续对矩阵: E_current = E[0:p-1], E_next = E[1:p]
    E_current = E[:-1, :]  # (p-1, embed_dim)
    E_next = E[1:, :]      # (p-1, embed_dim)

    # 向量化计算最优旋转矩阵 R_t
    R_t = compute_optimal_rotation_vectorized(E_current, E_next)

    # 计算残差: ||E_next - R_t * E_current||²_F
    # 使用向量化操作: E_next - E_current @ R_t^T
    predicted = E_current @ R_t.T  # (p-1, embed_dim)
    residuals = E_next - predicted
    residual_norm_squared = np.sum(residuals ** 2)  # Frobenius norm squared

    # 计算分母: Σ||E_{x+1}||²_F
    denominator = np.sum(E_next ** 2)

    # 相对残差 ε_R(t)
    if denominator > 1e-10:
        epsilon_R = residual_norm_squared / denominator
    else:
        epsilon_R = 0.0

    # 计算正交性偏差 δ_orth(t) = ||R_t^T @ R_t - I||_F
    orthogonality_deviation = np.linalg.norm(R_t.T @ R_t - np.eye(R_t.shape[0]), 'fro')

    # 额外指标：环闭合残差（从 p-1 到 0）
    E_last = E[-1:, :]  # (1, embed_dim)
    E_first = E[:1, :]  # (1, embed_dim)
    closure_residual = np.sum((E_first - E_last @ R_t.T) ** 2)

    return {
        'epsilon_R': float(epsilon_R),
        'delta_orth': float(orthogonality_deviation),
        'closure_residual': float(closure_residual),
    }


def extract_group_representation_from_checkpoints(checkpoint_dir, output_file, p=97):
    """
    从所有 checkpoint 文件中提取群表示指标

    Args:
        checkpoint_dir: checkpoint 目录
        output_file: 输出 CSV 文件
        p: 模数
    """
    # 获取所有 checkpoint 文件并按步数排序
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

            # 提取嵌入矩阵 W_E: (vocab_size, embed_dim)
            W_E = checkpoint['model_state_dict']['embedding.weight'].numpy()

            # 计算群表示指标
            metrics = compute_group_representation_metrics(W_E, p)

            result = {
                'step': step,
                'train_loss': float(checkpoint['train_loss']),
                'train_acc': float(checkpoint['train_acc']),
                'test_loss': float(checkpoint['test_loss']),
                'test_acc': float(checkpoint['test_acc']),
                **metrics
            }

            results.append(result)

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

        # 打印统计信息
        print(f"\n  最终指标:")
        print(f"    ε_R(t) = {results[-1]['epsilon_R']:.6f}")
        print(f"    δ_orth(t) = {results[-1]['delta_orth']:.6f}")
        print(f"    环闭合残差 = {results[-1]['closure_residual']:.6f}")

        return True
    else:
        print("  没有可保存的数据")
        return False


def main():
    """主函数 - 支持四种模运算"""
    parser = argparse.ArgumentParser(description='群论表示分析')
    parser.add_argument('--operation', type=str, default='x+y',
                        choices=['x+y', 'x-y', 'x*y', 'x_div_y', 'all'],
                        help='模运算类型')
    parser.add_argument('--p', type=int, default=97, help='模数')

    args = parser.parse_args()

    # 运算配置
    operations = {
        'x+y': {
            'name': 'Addition',
            'checkpoint_dir': '/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints',
            'output_file': '/root/data1/zjj/Grokking_Formulation/data/x+y/group_representation.csv'
        },
        'x-y': {
            'name': 'Subtraction',
            'checkpoint_dir': '/root/data1/zjj/Grokking_Formulation/data/x-y/checkpoints',
            'output_file': '/root/data1/zjj/Grokking_Formulation/data/x-y/group_representation.csv'
        },
        'x*y': {
            'name': 'Multiplication',
            'checkpoint_dir': '/root/data1/zjj/Grokking_Formulation/data/x*y/checkpoints',
            'output_file': '/root/data1/zjj/Grokking_Formulation/data/x*y/group_representation.csv'
        },
        'x_div_y': {
            'name': 'Division',
            'checkpoint_dir': '/root/data1/zjj/Grokking_Formulation/data/x_div_y/checkpoints',
            'output_file': '/root/data1/zjj/Grokking_Formulation/data/x_div_y/group_representation.csv'
        },
    }

    if args.operation == 'all':
        ops_to_process = list(operations.keys())
    else:
        ops_to_process = [args.operation]

    for op in ops_to_process:
        op_config = operations[op]
        checkpoint_dir = op_config['checkpoint_dir']
        output_file = op_config['output_file']

        print(f"\n{'='*60}")
        print(f"分析: {op_config['name']} ({op})")
        print(f"{'='*60}")
        print(f"Checkpoint 目录: {checkpoint_dir}")
        print(f"输出文件: {output_file}")
        print(f"{'='*60}")

        if not os.path.exists(checkpoint_dir):
            print(f"  跳过: 目录不存在 - {checkpoint_dir}")
            continue

        extract_group_representation_from_checkpoints(checkpoint_dir, output_file, args.p)

    print(f"\n{'='*60}")
    print("全部完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
