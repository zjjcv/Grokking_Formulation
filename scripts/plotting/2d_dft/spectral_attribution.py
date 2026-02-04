#!/usr/bin/env python3
"""
统一频谱归因绘图 - 支持所有四种运算
子图分开保存

使用方法:
    python spectral_attribution.py --operation x+y
    python spectral_attribution.py --all
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
        'low_freq': '#2ecc71', 'high_freq': '#e67e22', 'grokking': '#f1c40f',
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


def load_attribution_data(data_file):
    """加载频谱归因数据"""
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


def plot_low_high_ratio(steps, low_high_ratio, test_accs, output_dir, grokking_start=30000):
    """绘制低频/高频比率"""
    fig, ax = plt.subplots(figsize=(10, 6))

    line1, = ax.plot(steps, test_accs, color=COLORS['test_acc'],
                     linewidth=2.0, alpha=0.7, label='Test Acc')

    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Accuracy', fontsize=FONTS['label']['size'], color=COLORS['test_acc'])
    ax.tick_params(axis='y', labelcolor=COLORS['test_acc'], labelsize=FONTS['tick']['size'])
    ax.tick_params(axis='x', labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    ax2 = ax.twinx()
    line2, = ax2.plot(steps, low_high_ratio, color=COLORS['low_freq'],
                      linewidth=2.5, alpha=0.9, label='Low/High Ratio')

    ax2.set_ylabel('Low/High Frequency Ratio', fontsize=FONTS['label']['size'], color=COLORS['low_freq'])
    ax2.tick_params(axis='y', labelcolor=COLORS['low_freq'], labelsize=FONTS['tick']['size'])

    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='center left', fontsize=FONTS['legend']['size']-1)

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'low_high_ratio')
    plt.close()


def plot_frequency_band_evolution(steps, low_freq, high_freq, output_dir, grokking_start=30000):
    """绘制频率带演化"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, low_freq, color=COLORS['low_freq'],
            linewidth=2.0, label='Low Frequency (k≤5)')
    ax.plot(steps, high_freq, color=COLORS['high_freq'],
            linewidth=2.0, label='High Frequency (k>5)')

    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Frequency Attribution', fontsize=FONTS['label']['size'])
    ax.tick_params(labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTS['legend']['size'])

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'frequency_band_evolution')
    plt.close()


def plot_spatial_heatmap(data, output_dir, max_freq=20):
    """绘制频谱归因热力图"""
    # 提取频率归因数据
    steps = data['steps'][::50]  # 每50个点取一个，避免过密
    freq_data = []
    freq_keys = [f'freq_{k}_attribution' for k in range(max_freq)]

    for step_idx in range(0, len(data['steps']), 50):
        row = []
        for k in range(max_freq):
            key = f'freq_{k}_attribution'
            row.append(data[key][step_idx])
        freq_data.append(row)

    freq_data = np.array(freq_data).T

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(freq_data, aspect='auto', cmap='viridis', interpolation='nearest')

    ax.set_xlabel('Training Step (downsampled)', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Frequency Index', fontsize=FONTS['label']['size'])
    ax.set_yticks(range(0, max_freq, 5))
    ax.set_yticklabels(range(0, max_freq, 5))
    ax.tick_params(labelsize=FONTS['tick']['size'])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Frequency Attribution', fontsize=FONTS['label']['size']-1)

    save_figure(fig, output_dir, 'spectral_attribution_heatmap')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='绘制频谱归因图')
    parser.add_argument('--operation', type=str, default='x+y',
                        choices=['x+y', 'x-y', 'x*y', 'x_div_y', 'all'],
                        help='模运算类型')

    args = parser.parse_args()
    operations = [args.operation] if args.operation != 'all' else list(OPERATIONS.keys())

    for op_key in operations:
        print(f"\n{'='*60}")
        print(f"绘图: {OPERATIONS[op_key]['name']} ({op_key})")
        print(f"{'='*60}")

        data_file = f"{OPERATIONS[op_key]['data_dir']}/spectral_attribution.csv"
        output_dir = get_figures_dir(op_key)

        if not os.path.exists(data_file):
            print(f"  跳过: 数据文件不存在 - {data_file}")
            continue

        print(f"  加载数据: {data_file}")
        data = load_attribution_data(data_file)
        print(f"  数据点数: {len(data['steps'])}")

        print(f"  生成图表...")
        os.makedirs(output_dir, exist_ok=True)

        plot_low_high_ratio(data['steps'], data['low_high_ratio'], data['test_accs'], output_dir)
        print(f"    已保存: low_high_ratio")

        plot_frequency_band_evolution(data['steps'], data['low_freq_attribution'], data['high_freq_attribution'], output_dir)
        print(f"    已保存: frequency_band_evolution")

        plot_spatial_heatmap(data, output_dir)
        print(f"    已保存: spectral_attribution_heatmap")


if __name__ == "__main__":
    main()
