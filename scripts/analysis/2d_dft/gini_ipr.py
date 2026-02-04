#!/usr/bin/env python3
"""
统一 2D Gini 系数和 IPR 分析 - 支持所有四种运算

使用方法:
    python gini_ipr.py --operation x+y    # 加法
    python gini_ipr.py --operation x-y    # 减法
    python gini_ipr.py --operation x*y    # 乘法
    python gini_ipr.py --operation x_div_y  # 除法
    python gini_ipr.py --all              # 所有运算
"""

import os
import csv
import argparse
import torch
import numpy as np

# 添加父目录到路径以导入配置
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in __file__:
    import sys
    sys.path.insert(0, sys_path)

try:
    from lib.config import OPERATIONS, get_checkpoint_dir, get_metric_file
except ImportError:
    # 如果无法导入配置，使用默认路径
    OPERATIONS = {
        'x+y': {'name': 'Addition', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x+y'},
        'x-y': {'name': 'Subtraction', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x-y'},
        'x*y': {'name': 'Multiplication', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x*y'},
        'x_div_y': {'name': 'Division', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x_div_y'},
    }
    def get_checkpoint_dir(op): return f"{OPERATIONS[op]['data_dir']}/checkpoints"
    def get_metric_file(op): return f"{OPERATIONS[op]['data_dir']}/metric.csv"


def compute_gini_coefficient(values):
    """计算基尼系数"""
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
    """计算逆参与率"""
    values = np.array(values).flatten()
    values = np.abs(values)
    total = np.sum(values)
    if total == 0:
        return len(values)
    probs = values / total
    ipr = 1.0 / np.sum(probs ** 2)
    return ipr


def extract_2d_metrics_from_checkpoints(checkpoint_dir, output_file, p=97):
    """从所有 checkpoint 文件中提取嵌入矩阵并计算 2D DFT 后的 Gini 系数和 IPR"""
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
            embedding_matrix = checkpoint['model_state_dict']['embedding.weight'].numpy()

            # 只取前 p 行
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
                'train_loss': float(checkpoint['train_loss']),
                'train_acc': float(checkpoint['train_acc']),
                'test_loss': float(checkpoint['test_loss']),
                'test_acc': float(checkpoint['test_acc']),
                'gini': float(gini),
                'ipr': float(ipr)
            }

            results.append(result)

            if step % 10000 == 0:
                print(f"    Step {step}: Gini={gini:.4f}, IPR={ipr:.4f}")

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
        return True
    else:
        print("  没有可保存的数据")
        return False


def main():
    parser = argparse.ArgumentParser(description='计算 2D 频域 Gini 系数和 IPR')
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
        output_file = f"{OPERATIONS[op_key]['data_dir']}/gini_ip_2d.csv"

        if not os.path.exists(checkpoint_dir):
            print(f"  跳过: 检查点目录不存在 - {checkpoint_dir}")
            continue

        extract_2d_metrics_from_checkpoints(checkpoint_dir, output_file, args.p)


if __name__ == "__main__":
    main()
