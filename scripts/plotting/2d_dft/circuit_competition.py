#!/usr/bin/env python3
"""
统一电路竞争绘图 - 支持所有四种运算
子图分开保存

使用方法:
    python circuit_competition.py --operation x+y
    python circuit_competition.py --all
"""

import os
import sys
import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np

sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in __file__:
    sys.path.insert(0, sys_path)

try:
    from lib.config import (
        OPERATIONS, COLORS, FONTS,
        get_figures_dir, add_grokking_region, add_crossover_line, save_figure
    )
except ImportError:
    COLORS = {
        'train_acc': '#3498db', 'test_acc': '#e74c3c',
        'memory': '#c0392b', 'algorithm': '#2980b9', 'residual': '#7f8c8d',
        'crossover': '#9b59b6', 'grokking': '#f1c40f',
    }
    FONTS = {'label': {'size': 12}, 'tick': {'size': 10}, 'legend': {'size': 10}}
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
    def add_crossover_line(ax, x_pos, label=None):
        ax.axvline(x=x_pos, color=COLORS['crossover'], linestyle='--', linewidth=2, alpha=0.8, label=label)
    def save_figure(fig, output_dir, filename):
        os.makedirs(output_dir, exist_ok=True)
        for fmt in ['png', 'pdf']:
            filepath = os.path.join(output_dir, f'{filename}.{fmt}')
            fig.savefig(filepath, bbox_inches='tight', dpi=300 if fmt == 'png' else None)


def load_circuit_data(data_file):
    """加载电路竞争数据"""
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


def find_crossover(steps, ratios):
    """寻找 crossover 点"""
    for i in range(1, len(ratios)):
        if ratios[i-1] < 1.0 and ratios[i] >= 1.0:
            return steps[i]
    return None


def plot_energy_stack_WE(steps, data, output_dir, grokking_start=30000):
    """绘制 W_E 能量堆叠图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    memo = np.array(data['W_E_memo_energy'])
    fourier = np.array(data['W_E_fourier_energy'])
    residual = np.array(data['W_E_residual_energy'])

    ax.fill_between(steps, 0, memo, alpha=0.7, color=COLORS['memory'], label='Memory')
    ax.fill_between(steps, memo, memo + fourier, alpha=0.7, color=COLORS['algorithm'], label='Algorithm')
    ax.fill_between(steps, memo + fourier, memo + fourier + residual, alpha=0.3, color=COLORS['residual'], label='Residual')

    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Energy (W_E)', fontsize=FONTS['label']['size'])
    ax.tick_params(labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTS['legend']['size'])

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'energy_stack_WE')
    plt.close()


def plot_energy_stack_WU(steps, data, output_dir, grokking_start=30000):
    """绘制 W_U 能量堆叠图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    memo = np.array(data['W_U_memo_energy'])
    fourier = np.array(data['W_U_fourier_energy'])
    residual = np.array(data['W_U_residual_energy'])

    ax.fill_between(steps, 0, memo, alpha=0.7, color=COLORS['memory'], label='Memory')
    ax.fill_between(steps, memo, memo + fourier, alpha=0.7, color=COLORS['algorithm'], label='Algorithm')
    ax.fill_between(steps, memo + fourier, memo + fourier + residual, alpha=0.3, color=COLORS['residual'], label='Residual')

    # 标记 crossover
    crossover = find_crossover(steps, data['W_U_competition_ratio'])
    if crossover:
        add_crossover_line(ax, crossover, f'Crossover (Step {crossover})')

    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Energy (W_U)', fontsize=FONTS['label']['size'])
    ax.tick_params(labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTS['legend']['size'])

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'energy_stack_WU')
    plt.close()


def plot_competition_ratio(steps, data, output_dir, grokking_start=30000):
    """
    绘制竞争比率 + 训练/测试准确率

    双 y 轴：
    - 左轴：准确率（训练 + 测试）
    - 右轴：算法/记忆竞争比率（绿色）
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    train_accs = data['train_accs']
    test_accs = data['test_accs']
    ratios = data['W_U_competition_ratio']

    # 定义竞争比率的绿色（好看的翠绿色）
    ratio_color = '#27ae60'  # 翠绿色

    # 左 y 轴：准确率
    line_train, = ax.plot(steps, train_accs, color=COLORS['train_acc'],
                          linewidth=2.0, alpha=0.8, label='Train Acc')
    line_test, = ax.plot(steps, test_accs, color=COLORS['test_acc'],
                          linewidth=2.5, alpha=0.9, label='Test Acc')

    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Accuracy', fontsize=FONTS['label']['size'])
    ax.tick_params(axis='y', labelsize=FONTS['tick']['size'])
    ax.tick_params(axis='x', labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # 右 y 轴：竞争比率（绿色）
    ax2 = ax.twinx()
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    crossover = find_crossover(steps, ratios)
    if crossover:
        add_crossover_line(ax2, crossover, f'Crossover ({crossover})')

    line_ratio, = ax2.plot(steps, ratios, color=ratio_color,
                             linewidth=3.0, alpha=1.0, label='Algorithm/Memory Ratio')

    # 填充背景显示主导电路（使用更淡的绿色）
    ax2.fill_between(steps, 0, ratios, where=np.array(ratios) < 1,
                     alpha=0.15, color=COLORS['memory'], label='Memory Dominated')
    ax2.fill_between(steps, 0, ratios, where=np.array(ratios) >= 1,
                     alpha=0.15, color=ratio_color, label='Algorithm Dominated')

    ax2.set_ylabel('Algorithm / Memory Ratio', fontsize=FONTS['label']['size'],
                    color=ratio_color, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=ratio_color, labelsize=FONTS['tick']['size'])

    # 图例：合并左右 y 轴的线条
    lines = [line_train, line_test, line_ratio]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='center left', fontsize=FONTS['legend']['size'])

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'competition_ratio_with_acc')
    plt.close()


def plot_energy_comparison_WU(steps, data, output_dir, grokking_start=30000):
    """绘制 W_U 能量对比"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, data['W_U_memo_energy'], color=COLORS['memory'],
            linewidth=2.0, label='Memory Energy')
    ax.plot(steps, data['W_U_fourier_energy'], color=COLORS['algorithm'],
            linewidth=2.0, label='Algorithm Energy')

    crossover = find_crossover(steps, data['W_U_competition_ratio'])
    if crossover:
        add_crossover_line(ax, crossover)

    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Energy', fontsize=FONTS['label']['size'])
    ax.tick_params(labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTS['legend']['size'])

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'energy_comparison_WU')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='绘制电路竞争图')
    parser.add_argument('--operation', type=str, default='x+y',
                        choices=['x+y', 'x-y', 'x*y', 'x_div_y', 'all'],
                        help='模运算类型')

    args = parser.parse_args()
    operations = [args.operation] if args.operation != 'all' else list(OPERATIONS.keys())

    for op_key in operations:
        print(f"\n{'='*60}")
        print(f"绘图: {OPERATIONS[op_key]['name']} ({op_key})")
        print(f"{'='*60}")

        data_file = f"{OPERATIONS[op_key]['data_dir']}/circuit_Competition.csv"
        output_dir = get_figures_dir(op_key)

        if not os.path.exists(data_file):
            print(f"  跳过: 数据文件不存在 - {data_file}")
            continue

        print(f"  加载数据: {data_file}")
        data = load_circuit_data(data_file)
        print(f"  数据点数: {len(data['steps'])}")

        print(f"  生成图表...")
        os.makedirs(output_dir, exist_ok=True)

        plot_energy_stack_WE(data['steps'], data, output_dir)
        print(f"    已保存: energy_stack_WE")

        plot_energy_stack_WU(data['steps'], data, output_dir)
        print(f"    已保存: energy_stack_WU")

        plot_competition_ratio(data['steps'], data, output_dir)
        print(f"    已保存: competition_ratio_with_acc (含训练+测试准确率)")

        plot_energy_comparison_WU(data['steps'], data, output_dir)
        print(f"    已保存: energy_comparison_WU")


if __name__ == "__main__":
    main()
