#!/usr/bin/env python3
"""
统一 QK 电路 2D 频域绘图 - 支持所有四种运算
子图单独保存

使用方法:
    python qk_circut.py --operation x+y
    python qk_circut.py --all
"""

import os
import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
    COLORS = {
        'train_acc': '#3498db', 'test_acc': '#e74c3c',
        'grokking': '#f1c40f', 'low_freq': '#2ecc71', 'high_freq': '#e67e22',
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


def load_qk_circuit_data(data_file):
    """加载 QK 电路数据"""
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


def plot_dc_component(steps, dc_component, train_accs, test_accs, output_dir, grokking_start=30000):
    """绘制 DC 分量随训练的变化（包含训练准确率）"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制训练和测试准确率
    line_train, = ax.plot(steps, train_accs, color=COLORS['train_acc'],
                          linewidth=2.0, alpha=0.7, label='Train Acc')
    line_test, = ax.plot(steps, test_accs, color=COLORS['test_acc'],
                         linewidth=2.0, alpha=0.7, label='Test Acc')
    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Accuracy', fontsize=FONTS['label']['size'])
    ax.tick_params(axis='y', labelsize=FONTS['tick']['size'])
    ax.tick_params(axis='x', labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # 右y轴：DC分量
    ax2 = ax.twinx()
    line_dc, = ax2.plot(steps, dc_component, color=COLORS['low_freq'],
                        linewidth=2.5, alpha=0.9, label='DC Component')

    ax2.set_ylabel('DC Component', fontsize=FONTS['label']['size'], color=COLORS['low_freq'])
    ax2.tick_params(axis='y', labelcolor=COLORS['low_freq'], labelsize=FONTS['tick']['size'])

    # 图例（合并左右轴）
    lines = [line_train, line_test, line_dc]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='center left', fontsize=FONTS['legend']['size']-1)

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'qk_dc_component')
    plt.close()


def plot_frequency_energy(steps, low_freq, high_freq, test_accs, output_dir, grokking_start=30000):
    """绘制低频/高频能量对比"""
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
    line2, = ax2.plot(steps, low_freq, color=COLORS['low_freq'],
                      linewidth=2.0, alpha=0.8, label='Low Freq Energy')
    line3, = ax2.plot(steps, high_freq, color=COLORS['high_freq'],
                      linewidth=2.0, alpha=0.8, label='High Freq Energy')

    ax2.set_ylabel('Frequency Energy', fontsize=FONTS['label']['size'])
    ax2.tick_params(labelsize=FONTS['tick']['size'])

    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='center left', fontsize=FONTS['legend']['size']-1)

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'qk_frequency_energy')
    plt.close()


def plot_spectral_entropy(steps, spectral_entropy, train_accs, test_accs, output_dir, grokking_start=30000):
    """绘制频谱熵（包含训练准确率）"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制训练和测试准确率
    line_train, = ax.plot(steps, train_accs, color=COLORS['train_acc'],
                          linewidth=2.0, alpha=0.7, label='Train Acc')
    line_test, = ax.plot(steps, test_accs, color=COLORS['test_acc'],
                         linewidth=2.0, alpha=0.7, label='Test Acc')
    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Accuracy', fontsize=FONTS['label']['size'])
    ax.tick_params(axis='y', labelsize=FONTS['tick']['size'])
    ax.tick_params(axis='x', labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # 右y轴：频谱熵
    ax2 = ax.twinx()
    line_entropy, = ax2.plot(steps, spectral_entropy, color='purple',
                              linewidth=2.5, alpha=0.9, label='Spectral Entropy')

    ax2.set_ylabel('Spectral Entropy', fontsize=FONTS['label']['size'], color='purple')
    ax2.tick_params(axis='y', labelcolor='purple', labelsize=FONTS['tick']['size'])

    # 图例（合并左右轴）
    lines = [line_train, line_test, line_entropy]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='center left', fontsize=FONTS['legend']['size']-1)

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'qk_spectral_entropy')
    plt.close()


def plot_magnitude_stats(steps, mag_mean, mag_std, output_dir, grokking_start=30000):
    """绘制幅度统计量"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # 均值
    ax = axes[0]
    ax.plot(steps, mag_mean, color=COLORS['low_freq'], linewidth=2)
    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size']-1)
    ax.set_ylabel('Mean Magnitude', fontsize=FONTS['label']['size']-1)
    ax.tick_params(labelsize=FONTS['tick']['size']-1)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    add_grokking_region(ax, steps, grokking_start)

    # 标准差
    ax = axes[1]
    ax.plot(steps, mag_std, color=COLORS['high_freq'], linewidth=2)
    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size']-1)
    ax.set_ylabel('Magnitude Std', fontsize=FONTS['label']['size']-1)
    ax.tick_params(labelsize=FONTS['tick']['size']-1)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    add_grokking_region(ax, steps, grokking_start)

    plt.tight_layout()
    save_figure(fig, output_dir, 'qk_magnitude_stats')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='绘制 QK 电路 2D 频域图')
    parser.add_argument('--operation', type=str, default='x+y',
                        choices=['x+y', 'x-y', 'x*y', 'x_div_y', 'all'],
                        help='模运算类型')

    args = parser.parse_args()

    operations = [args.operation] if args.operation != 'all' else list(OPERATIONS.keys())

    for op_key in operations:
        print(f"\n{'='*60}")
        print(f"绘图: {OPERATIONS[op_key]['name']} ({op_key})")
        print(f"{'='*60}")

        data_file = f"{OPERATIONS[op_key]['data_dir']}/qk_circut.csv"
        output_dir = get_figures_dir(op_key)

        if not os.path.exists(data_file):
            print(f"  跳过: 数据文件不存在 - {data_file}")
            continue

        print(f"  加载数据: {data_file}")
        data = load_qk_circuit_data(data_file)
        print(f"  数据点数: {len(data['steps'])}")

        print(f"  生成图表...")
        os.makedirs(output_dir, exist_ok=True)

        plot_dc_component(data['steps'], data['dc_component'], data['train_accs'], data['test_accs'], output_dir)
        print(f"    已保存: qk_dc_component")

        plot_frequency_energy(data['steps'], data['low_freq_energy'], data['high_freq_energy'], data['test_accs'], output_dir)
        print(f"    已保存: qk_frequency_energy")

        plot_spectral_entropy(data['steps'], data['spectral_entropy'], data['train_accs'], data['test_accs'], output_dir)
        print(f"    已保存: qk_spectral_entropy")

        plot_magnitude_stats(data['steps'], data['magnitude_mean'], data['magnitude_std'], output_dir)
        print(f"    已保存: qk_magnitude_stats")


if __name__ == "__main__":
    main()
