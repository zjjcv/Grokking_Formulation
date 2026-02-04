#!/usr/bin/env python3
"""
有效秩 (Effective Rank) 绘图 - 支持所有四种运算

使用 effective_rank.csv 和 metric.csv 绘制 Acc 和有效秩的变化

使用方法:
    python effective_rank.py --operation x+y
    python effective_rank.py --all
"""

import os
import sys
import csv
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

# 添加父目录到路径
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in __file__:
    sys.path.insert(0, sys_path)

try:
    from lib.config import (
        OPERATIONS, COLORS, FONTS,
        get_figures_dir, add_grokking_region, save_figure
    )
except ImportError:
    # 统一配色方案 - 美观配色
    COLORS = {
        'train_acc': '#3498db',      # 蓝色 - 训练准确率
        'test_acc': '#e74c3c',       # 红色 - 测试准确率
        'erank': '#8e44ad',          # 紫色 - 有效秩
        'grokking': '#f1c40f',       # 黄色 - Grokking区域
        'W_E': '#1abc9c',            # 青色 - Embedding
        'W_Q': '#3498db',            # 蓝色 - Query
        'W_K': '#9b59b6',            # 紫色 - Key
        'W_V': '#e67e22',            # 橙色 - Value
        'W_O': '#16a085',            # 绿色 - Output
        'W_1': '#d35400',            # 深橙 - FFN1
        'W_2': '#c0392b',            # 深红 - FFN2
    }
    FONTS = {'label': {'size': 12}, 'tick': {'size': 10}}
    OPERATIONS = {
        'x+y': {'name': 'Addition', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x+y'},
        'x-y': {'name': 'Subtraction', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x-y'},
        'x*y': {'name': 'Multiplication', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x*y'},
        'x_div_y': {'name': 'Division', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x_div_y'},
    }
    def get_figures_dir(op): return f"/root/data1/zjj/Grokking_Formulation/experiments/figures/{op}"
    def add_grokking_region(ax, steps, start_step=30000):
        max_step = max(steps) if hasattr(steps, '__iter__') else steps
        ax.axvspan(start_step, max_step, alpha=0.1, color=COLORS['grokking'])
    def save_figure(fig, output_dir, filename):
        os.makedirs(output_dir, exist_ok=True)
        for fmt in ['png', 'pdf']:
            filepath = os.path.join(output_dir, f'{filename}.{fmt}')
            fig.savefig(filepath, bbox_inches='tight', dpi=300 if fmt == 'png' else None)


def load_effective_rank_data(data_file):
    """加载有效秩数据"""
    data = {'steps': [], 'train_accs': [], 'test_accs': []}
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['steps'].append(int(row['step']))
            data['train_accs'].append(float(row['train_acc']))
            data['test_accs'].append(float(row['test_acc']))
            for key in row:
                if key not in data:
                    data[key] = []
                data[key].append(float(row[key]))
    return data


def plot_erank_with_accuracy(steps, train_accs, test_accs, erank_values, output_dir, matrix_name, grokking_start=30000):
    """绘制单个矩阵的有效秩与准确率（双y轴）"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 左 y 轴：准确率（训练 + 测试）
    line_train, = ax.plot(steps, train_accs, color=COLORS['train_acc'],
                          linewidth=1.5, alpha=0.6, label='Train Acc')
    line_test, = ax.plot(steps, test_accs, color=COLORS['test_acc'],
                         linewidth=2.0, alpha=0.9, label='Test Acc')
    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Accuracy', fontsize=FONTS['label']['size'])
    ax.tick_params(axis='y', labelsize=FONTS['tick']['size'])
    ax.tick_params(axis='x', labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # 右 y 轴：有效秩
    ax2 = ax.twinx()
    # 使用矩阵对应的颜色
    erank_color = COLORS.get(matrix_name, COLORS['erank'])
    line_erank, = ax2.plot(steps, erank_values, color=erank_color,
                           linewidth=2.5, alpha=0.9, label=f'{matrix_name} Effective Rank')
    ax2.set_ylabel('Effective Rank', fontsize=FONTS['label']['size'], color=erank_color, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=erank_color, labelsize=FONTS['tick']['size'])

    # 图例：合并左右 y 轴的线条
    lines = [line_train, line_test, line_erank]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='best', fontsize=FONTS['label']['size']-1)

    add_grokking_region(ax, steps, grokking_start)

    filename = f'erank_{matrix_name.lower().replace("_", "")}'
    save_figure(fig, output_dir, filename)
    plt.close()


def plot_all_eranks(data, output_dir, grokking_start=30000):
    """绘制所有矩阵的有效秩"""
    matrices = ['W_E', 'W_Q', 'W_K', 'W_V', 'W_O', 'W_1', 'W_2']

    for matrix in matrices:
        erank_key = f'{matrix}_erank'
        if erank_key in data:
            plot_erank_with_accuracy(data['steps'], data['train_accs'], data['test_accs'],
                                     data[erank_key], output_dir, matrix, grokking_start)
            print(f"    已保存: erank_{matrix.lower()}")


def plot_layer_comparison(data, output_dir, grokking_start=30000):
    """绘制同层不同矩阵的有效秩对比（无标题）"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # 左 y 轴：准确率（训练 + 测试）
    line_train, = ax.plot(data['steps'], data['train_accs'], color=COLORS['train_acc'],
                          linewidth=1.5, alpha=0.5, label='Train Acc')
    line_test, = ax.plot(data['steps'], data['test_accs'], color=COLORS['test_acc'],
                         linewidth=2.0, alpha=0.7, label='Test Acc')
    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Accuracy', fontsize=FONTS['label']['size'])
    ax.tick_params(axis='y', labelsize=FONTS['tick']['size'])
    ax.tick_params(axis='x', labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # 右 y 轴：有效秩
    ax2 = ax.twinx()

    # 绘制所有矩阵的有效秩，使用定义的颜色
    layer1_matrices = ['W_E', 'W_Q', 'W_K', 'W_V', 'W_O', 'W_1', 'W_2']

    lines = [line_train, line_test]
    labels = [line_train.get_label(), line_test.get_label()]

    for matrix in layer1_matrices:
        erank_key = f'{matrix}_erank'
        if erank_key in data:
            erank_color = COLORS.get(matrix, COLORS['erank'])
            line, = ax2.plot(data['steps'], data[erank_key],
                           color=erank_color, linewidth=2.0, alpha=0.85,
                           label=f'{matrix}')
            lines.append(line)
            labels.append(line.get_label())

    ax2.set_ylabel('Effective Rank', fontsize=FONTS['label']['size'])
    ax2.tick_params(labelsize=FONTS['tick']['size'])

    ax.legend(lines, labels, loc='center left', bbox_to_anchor=(1.12, 0.5),
              fontsize=FONTS['label']['size']-2, frameon=True, shadow=True)

    add_grokking_region(ax, data['steps'], grokking_start)

    save_figure(fig, output_dir, 'erank_all_matrices_comparison')
    plt.close()

    print(f"    已保存: erank_all_matrices_comparison")


def main():
    parser = argparse.ArgumentParser(description='绘制有效秩分析图')
    parser.add_argument('--operation', type=str, default='x+y',
                        choices=['x+y', 'x-y', 'x*y', 'x_div_y', 'all'],
                        help='模运算类型')

    args = parser.parse_args()

    operations = [args.operation] if args.operation != 'all' else list(OPERATIONS.keys())

    for op_key in operations:
        print(f"\n{'='*60}")
        print(f"绘图: {OPERATIONS[op_key]['name']} ({op_key})")
        print(f"{'='*60}")

        data_file = f"{OPERATIONS[op_key]['data_dir']}/effective_rank.csv"
        output_dir = get_figures_dir(op_key)

        if not os.path.exists(data_file):
            print(f"  跳过: 数据文件不存在 - {data_file}")
            continue

        print(f"  加载数据: {data_file}")
        data = load_effective_rank_data(data_file)
        print(f"  数据点数: {len(data['steps'])}")

        print(f"  生成图表...")
        os.makedirs(output_dir, exist_ok=True)

        plot_all_eranks(data, output_dir)
        plot_layer_comparison(data, output_dir)


if __name__ == "__main__":
    main()
