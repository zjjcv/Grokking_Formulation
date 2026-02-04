#!/usr/bin/env python3
"""
统一 2D 频域对齐绘图 - 支持所有四种运算
子图分开保存

使用方法:
    python f_alignment.py --operation x+y
    python f_alignment.py --all
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
        get_figures_dir, add_grokking_region, save_figure
    )
except ImportError:
    COLORS = {
        'train_acc': '#3498db', 'test_acc': '#e74c3c',
        'alignment': '#f39c12', 'grokking': '#f1c40f',
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
    def save_figure(fig, output_dir, filename):
        os.makedirs(output_dir, exist_ok=True)
        for fmt in ['png', 'pdf']:
            filepath = os.path.join(output_dir, f'{filename}.{fmt}')
            fig.savefig(filepath, bbox_inches='tight', dpi=300 if fmt == 'png' else None)


def load_alignment_data(data_file):
    """加载对齐数据"""
    steps, train_accs, test_accs, alignments, w_e_energies, w_u_energies = [], [], [], [], [], []
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            train_accs.append(float(row['train_acc']))
            test_accs.append(float(row['test_acc']))
            alignments.append(float(row['mean_2d_alignment']))
            w_e_energies.append(float(row['w_e_energy']))
            w_u_energies.append(float(row['w_u_energy']))
    return steps, train_accs, test_accs, alignments, w_e_energies, w_u_energies


def plot_alignment_with_accuracy(steps, alignments, train_accs, test_accs, output_dir, grokking_start=30000):
    """绘制对齐度与准确率（双 y 轴）"""
    fig, ax = plt.subplots(figsize=(10, 6))

    line1, = ax.plot(steps, train_accs, color=COLORS['train_acc'],
                     linewidth=1.5, alpha=0.6, label='Train Acc')
    line2, = ax.plot(steps, test_accs, color=COLORS['test_acc'],
                     linewidth=2.0, alpha=0.8, label='Test Acc')

    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Accuracy', fontsize=FONTS['label']['size'], color=COLORS['train_acc'])
    ax.tick_params(axis='y', labelcolor=COLORS['train_acc'], labelsize=FONTS['tick']['size'])
    ax.tick_params(axis='x', labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    ax2 = ax.twinx()
    line3, = ax2.plot(steps, alignments, color=COLORS['alignment'],
                      linewidth=2.5, alpha=0.9, label='2D Alignment')

    ax2.set_ylabel('2D Alignment (Cosine Similarity)', fontsize=FONTS['label']['size'], color=COLORS['alignment'])
    ax2.tick_params(axis='y', labelcolor=COLORS['alignment'], labelsize=FONTS['tick']['size'])

    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='center left', fontsize=FONTS['legend']['size']-1)

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'alignment_with_accuracy')
    plt.close()


def plot_alignment_evolution(steps, alignments, output_dir, grokking_start=30000):
    """绘制对齐度演化"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, alignments, color=COLORS['alignment'], linewidth=2.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline (0.5)')

    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('2D Alignment', fontsize=FONTS['label']['size'])
    ax.tick_params(labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTS['legend']['size'])

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'alignment_evolution')
    plt.close()


def plot_energy_comparison(steps, w_e_energies, w_u_energies, output_dir, grokking_start=30000):
    """绘制能量对比"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, w_e_energies, color=COLORS['train_acc'],
            linewidth=2.0, alpha=0.8, label='W_E Energy')
    ax.plot(steps, w_u_energies, color=COLORS['test_acc'],
            linewidth=2.0, alpha=0.8, label='W_U Energy')

    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Energy', fontsize=FONTS['label']['size'])
    ax.tick_params(labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTS['legend']['size'])

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'energy_comparison')
    plt.close()


def plot_alignment_vs_accuracy_scatter(alignments, test_accs, steps, output_dir):
    """绘制对齐度 vs 准确率散点图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(test_accs, alignments, c=steps, cmap='viridis', alpha=0.6, s=10)
    ax.set_xlabel('Test Accuracy', fontsize=FONTS['label']['size'])
    ax.set_ylabel('2D Alignment', fontsize=FONTS['label']['size'])
    ax.tick_params(labelsize=FONTS['tick']['size'])
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Training Step', fontsize=FONTS['label']['size']-1)

    save_figure(fig, output_dir, 'alignment_vs_accuracy_scatter')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='绘制 2D 频域对齐图')
    parser.add_argument('--operation', type=str, default='x+y',
                        choices=['x+y', 'x-y', 'x*y', 'x_div_y', 'all'],
                        help='模运算类型')

    args = parser.parse_args()
    operations = [args.operation] if args.operation != 'all' else list(OPERATIONS.keys())

    for op_key in operations:
        print(f"\n{'='*60}")
        print(f"绘图: {OPERATIONS[op_key]['name']} ({op_key})")
        print(f"{'='*60}")

        data_file = f"{OPERATIONS[op_key]['data_dir']}/f_alginment_2d.csv"
        output_dir = get_figures_dir(op_key)

        if not os.path.exists(data_file):
            print(f"  跳过: 数据文件不存在 - {data_file}")
            continue

        print(f"  加载数据: {data_file}")
        steps, train_accs, test_accs, alignments, w_e_energies, w_u_energies = load_alignment_data(data_file)
        print(f"  数据点数: {len(steps)}")

        print(f"  生成图表...")
        os.makedirs(output_dir, exist_ok=True)

        plot_alignment_with_accuracy(steps, alignments, train_accs, test_accs, output_dir)
        print(f"    已保存: alignment_with_accuracy")

        plot_alignment_evolution(steps, alignments, output_dir)
        print(f"    已保存: alignment_evolution")

        plot_energy_comparison(steps, w_e_energies, w_u_energies, output_dir)
        print(f"    已保存: energy_comparison")

        plot_alignment_vs_accuracy_scatter(alignments, test_accs, steps, output_dir)
        print(f"    已保存: alignment_vs_accuracy_scatter")


if __name__ == "__main__":
    main()
