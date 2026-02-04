#!/usr/bin/env python3
"""
统一傅里叶投影绘图 - 支持所有四种运算
子图分开保存

使用方法:
    python fourier_projection.py --operation x+y
    python fourier_projection.py --all
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
        'algorithm': '#2980b9', 'memory': '#c0392b', 'grokking': '#f1c40f',
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


def load_projection_data(data_file):
    """加载投影数据"""
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


def plot_l1l2_comparison(steps, data, output_dir, grokking_start=30000):
    """绘制 L1/L2 对比"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, data['W_E_spatial_l1l2'], color=COLORS['memory'],
            linewidth=2.0, alpha=0.8, label='W_E Spatial')
    ax.plot(steps, data['W_E_fourier_l1l2'], color=COLORS['algorithm'],
            linewidth=2.0, alpha=0.8, label='W_E Fourier')

    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('L1/L2 Ratio (W_E)', fontsize=FONTS['label']['size'])
    ax.tick_params(labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTS['legend']['size'])

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'l1l2_comparison_WE')
    plt.close()


def plot_gini_comparison(steps, data, output_dir, grokking_start=30000):
    """绘制 Gini 系数对比"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, data['W_E_spatial_gini'], color=COLORS['memory'],
            linewidth=2.0, alpha=0.8, label='W_E Spatial')
    ax.plot(steps, data['W_E_fourier_gini'], color=COLORS['algorithm'],
            linewidth=2.0, alpha=0.8, label='W_E Fourier')

    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Gini Coefficient (W_E)', fontsize=FONTS['label']['size'])
    ax.tick_params(labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTS['legend']['size'])

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'gini_comparison_WE')
    plt.close()


def plot_fourier_domains_comparison(steps, data, output_dir, grokking_start=30000):
    """绘制傅里叶域对比（W_E vs W_U）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # W_E
    ax = axes[0]
    ax.plot(steps, data['W_E_fourier_l1l2'], color=COLORS['train_acc'],
            linewidth=2.0, label='L1/L2')
    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size']-1)
    ax.set_ylabel('L1/L2 Ratio', fontsize=FONTS['label']['size']-1)
    ax.tick_params(labelsize=FONTS['tick']['size']-1)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTS['legend']['size']-1)
    add_grokking_region(ax, steps, grokking_start)

    # W_U
    ax = axes[1]
    ax.plot(steps, data['W_U_fourier_l1l2'], color=COLORS['test_acc'],
            linewidth=2.0, label='L1/L2')
    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size']-1)
    ax.set_ylabel('L1/L2 Ratio', fontsize=FONTS['label']['size']-1)
    ax.tick_params(labelsize=FONTS['tick']['size']-1)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTS['legend']['size']-1)
    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'fourier_domains_comparison')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='绘制傅里叶投影图')
    parser.add_argument('--operation', type=str, default='x+y',
                        choices=['x+y', 'x-y', 'x*y', 'x_div_y', 'all'],
                        help='模运算类型')

    args = parser.parse_args()
    operations = [args.operation] if args.operation != 'all' else list(OPERATIONS.keys())

    for op_key in operations:
        print(f"\n{'='*60}")
        print(f"绘图: {OPERATIONS[op_key]['name']} ({op_key})")
        print(f"{'='*60}")

        data_file = f"{OPERATIONS[op_key]['data_dir']}/fourier_projection_2d.csv"
        output_dir = get_figures_dir(op_key)

        if not os.path.exists(data_file):
            print(f"  跳过: 数据文件不存在 - {data_file}")
            continue

        print(f"  加载数据: {data_file}")
        data = load_projection_data(data_file)
        print(f"  数据点数: {len(data['steps'])}")

        print(f"  生成图表...")
        os.makedirs(output_dir, exist_ok=True)

        plot_l1l2_comparison(data['steps'], data, output_dir)
        print(f"    已保存: l1l2_comparison_WE")

        plot_gini_comparison(data['steps'], data, output_dir)
        print(f"    已保存: gini_comparison_WE")

        plot_fourier_domains_comparison(data['steps'], data, output_dir)
        print(f"    已保存: fourier_domains_comparison")


if __name__ == "__main__":
    main()
