#!/usr/bin/env python3
"""
统一 2D 频域对齐分析 - 支持所有四种运算

计算输入嵌入 $W_E$ 和输出嵌入 $W_U$ 在 2D 频域上的余弦相似度

使用方法:
    python f_alginment.py --operation x+y
    python f_alginment.py --all
"""

import os
import csv
import argparse
import torch
import numpy as np

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


def compute_cosine_similarity_2d(mat1, mat2):
    """计算两个 2D 矩阵的余弦相似度"""
    vec1 = mat1.flatten()
    vec2 = mat2.flatten()
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    similarity = dot_product / (norm1 * norm2)
    return similarity


def compute_2d_frequency_alignment(checkpoint_dir, output_file, p=97):
    """从所有 checkpoint 文件中计算 2D 频域对齐"""
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("checkpoint_step_") and f.endswith(".pt"):
            step = int(f.replace("checkpoint_step_", "").replace(".pt", ""))
            checkpoint_files.append((step, os.path.join(checkpoint_dir, f)))

    checkpoint_files.sort(key=lambda x: x[0])
    print(f"  找到 {len(checkpoint_files)} 个 checkpoint 文件")

    results = []

    for step, checkpoint_path in checkpoint_files:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 提取输入嵌入 W_E: (p+1, embed_dim)
            W_E = checkpoint['model_state_dict']['embedding.weight'].numpy()

            # 提取输出权重 W_U: (p, embed_dim)
            W_U = checkpoint['model_state_dict']['output.weight'].numpy()

            # 只取前 p 行
            W_E_p = W_E[:p, :]

            # 进行 2D DFT
            W_E_dft = np.fft.fft2(W_E_p)
            W_U_dft = np.fft.fft2(W_U)

            # 取幅度
            W_E_dft_mag = np.abs(W_E_dft)
            W_U_dft_mag = np.abs(W_U_dft)

            # 计算余弦相似度
            similarity = compute_cosine_similarity_2d(W_E_dft_mag, W_U_dft_mag)

            # 计算其他统计量
            w_e_energy = float(np.sum(W_E_dft_mag ** 2))
            w_u_energy = float(np.sum(W_U_dft_mag ** 2))
            dc_E = float(W_E_dft_mag[0, 0])
            dc_U = float(W_U_dft_mag[0, 0])

            result = {
                'step': step,
                'train_loss': float(checkpoint['train_loss']),
                'train_acc': float(checkpoint['train_acc']),
                'test_loss': float(checkpoint['test_loss']),
                'test_acc': float(checkpoint['test_acc']),
                'mean_2d_alignment': float(similarity),
                'w_e_energy': w_e_energy,
                'w_u_energy': w_u_energy,
                'dc_component_E': dc_E,
                'dc_component_U': dc_U,
            }

            results.append(result)

            if step % 10000 == 0:
                print(f"    Step {step}: 2D Alignment={similarity:.4f}")

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
        alignments = [r['mean_2d_alignment'] for r in results]
        print(f"  2D 频域对齐统计:")
        print(f"    初始值: {alignments[0]:.4f}")
        print(f"    最终值: {alignments[-1]:.4f}")
        print(f"    最小值: {np.min(alignments):.4f}")
        print(f"    最大值: {np.max(alignments):.4f}")
        print(f"    平均值: {np.mean(alignments):.4f}")
        return True
    else:
        print("  没有可保存的数据")
        return False


def main():
    parser = argparse.ArgumentParser(description='计算 2D 频域对齐')
    parser.add_argument('--operation', type=str, default='x+y',
                        choices=['x+y', 'x-y', 'x*y', 'x_div_y', 'all'],
                        help='模运算类型')
    parser.add_argument('--p', type=int, default=97, help='模数')

    args = parser.parse_args()

    operations = [args.operation] if args.operation != 'all' else list(OPERATIONS.keys())

    for op_key in operations:
        print(f"\n{'='*60}")
        print(f"分析: {OPERATIONS[op_key]['name']} ({op_key})")
        print(f"{'='*60}")

        checkpoint_dir = get_checkpoint_dir(op_key)
        output_file = f"{OPERATIONS[op_key]['data_dir']}/f_alginment_2d.csv"

        if not os.path.exists(checkpoint_dir):
            print(f"  跳过: 检查点目录不存在 - {checkpoint_dir}")
            continue

        compute_2d_frequency_alignment(checkpoint_dir, output_file, args.p)


if __name__ == "__main__":
    main()
