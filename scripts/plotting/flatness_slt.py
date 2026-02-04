#!/usr/bin/env python3
"""
平坦度分析绘图 (Flatness SLT Plot)

绘制 Acc 和有效学习系数 λ 随训练步数的变化

使用方法:
    python flatness_slt.py --operation x+y
    python flatness_slt.py --all
"""

import os
import sys
import csv
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# 配置字体
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False

# 添加父目录到路径
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

try:
    from lib.config import (
        OPERATIONS, COLORS, FONTS,
        get_figures_dir, add_grokking_region, save_figure
    )
except ImportError:
    COLORS = {
        'train_acc': '#3498db', 'test_acc': '#e74c3c',
        'lambda_eff': '#9b59b6', 'energy': '#e67e22',
        'grokking': '#f1c40f',
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
            print(f'  已保存: {filepath}')


def load_flatness_data(data_file):
    """加载平坦度分析数据"""
    data = {
        'steps': [],
        'train_accs': [],
        'test_accs': [],
        'lambda_eff': [],
    }

    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['steps'].append(int(row['step']))
            data['train_accs'].append(float(row['train_acc']))
            data['test_accs'].append(float(row['test_acc']))
            data['lambda_eff'].append(float(row['lambda_eff']))

            # 动态添加能量数据
            for key in row:
                if key.startswith('energy_beta_') and key not in data:
                    data[key] = []
                if key.startswith('energy_beta_'):
                    data[key].append(float(row[key]))

    return data


def plot_flatness_dual_axis(data, output_dir, grokking_start=30000):
    """绘制双y轴图：Acc + λ_eff"""

    steps = data['steps']
    test_accs = data['test_accs']
    lambda_eff = data['lambda_eff']

    # 1. Acc + λ_eff 双轴图
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左 y 轴：准确率
    line1, = ax1.plot(steps, test_accs, color=COLORS['test_acc'],
                      linewidth=2.0, alpha=0.9, label='Test Acc')
    ax1.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax1.set_ylabel('Accuracy', fontsize=FONTS['label']['size'], color=COLORS['test_acc'])
    ax1.tick_params(axis='y', labelcolor=COLORS['test_acc'], labelsize=FONTS['tick']['size'])
    ax1.tick_params(axis='x', labelsize=FONTS['tick']['size'])
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # 右 y 轴：有效 λ
    ax2 = ax1.twinx()
    line2, = ax2.plot(steps, lambda_eff, color=COLORS['lambda_eff'],
                      linewidth=2.5, alpha=0.9, label=r'$\lambda_{eff}$')
    ax2.set_ylabel('Effective Learning Coefficient', fontsize=FONTS['label']['size'],
                   color=COLORS['lambda_eff'])
    ax2.tick_params(axis='y', labelcolor=COLORS['lambda_eff'], labelsize=FONTS['tick']['size'])

    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='best', fontsize=FONTS['label']['size']-1)

    add_grokking_region(ax1, steps, grokking_start)

    save_figure(fig, output_dir, 'flatness_slt_acc_lambda')
    plt.close()

    # 2. λ_eff 单独图（带更详细的标注）
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, lambda_eff, color=COLORS['lambda_eff'],
            linewidth=2.5, alpha=0.9, label=r'$\lambda_{eff}$')
    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Effective Learning Coefficient', fontsize=FONTS['label']['size'])
    ax.tick_params(labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTS['label']['size']-1)

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'flatness_slt_lambda_only')
    plt.close()

    # 3. λ_eff 变化率图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 计算变化率
    lambda_diff = np.diff(lambda_eff)
    ax.plot(steps[1:], lambda_diff, color='#e67e22',
            linewidth=2.0, alpha=0.8, label=r'$\Delta\lambda_{eff}$')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('λ Change Rate', fontsize=FONTS['label']['size'])
    ax.tick_params(labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTS['label']['size']-1)

    add_grokking_region(ax, steps[1:], grokking_start)

    save_figure(fig, output_dir, 'flatness_slt_lambda_change_rate')
    plt.close()

    # 4. 不同 β 下的能量图（如果有）
    energy_keys = [k for k in data.keys() if k.startswith('energy_beta_')]
    if energy_keys:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 左 y 轴：准确率
        line1, = ax1.plot(steps, test_accs, color=COLORS['test_acc'],
                          linewidth=2.0, alpha=0.7, label='Test Acc')
        ax1.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
        ax1.set_ylabel('Accuracy', fontsize=FONTS['label']['size'], color=COLORS['test_acc'])
        ax1.tick_params(axis='y', labelcolor=COLORS['test_acc'], labelsize=FONTS['tick']['size'])
        ax1.tick_params(axis='x', labelsize=FONTS['tick']['size'])
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])

        # 右 y 轴：能量
        ax2 = ax1.twinx()
        colors_energy = plt.cm.coolwarm(np.linspace(0, 1, len(energy_keys)))

        lines = [line1]
        labels = [line1.get_label()]

        for i, key in enumerate(energy_keys):
            beta = key.split('_')[-1]
            line, = ax2.plot(steps, data[key], color=colors_energy[i],
                           linewidth=1.5, alpha=0.7, label=f'β={beta}')
            lines.append(line)
            labels.append(line.get_label())

        ax2.set_ylabel('Free Energy Estimate', fontsize=FONTS['label']['size'])
        ax2.tick_params(labelsize=FONTS['tick']['size'])

        ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(1.15, 0.5),
                   fontsize=FONTS['label']['size']-2)

        add_grokking_region(ax1, steps, grokking_start)

        plt.tight_layout()
        save_figure(fig, output_dir, 'flatness_slt_energy_by_beta')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='绘制平坦度分析图')
    parser.add_argument('--operation', type=str, default='all',
                        choices=['x+y', 'x-y', 'x*y', 'x_div_y', 'all'],
                        help='模运算类型')

    args = parser.parse_args()

    operations = [args.operation] if args.operation != 'all' else list(OPERATIONS.keys())

    for op_key in operations:
        print(f"\n{'='*60}")
        print(f"绘图: {OPERATIONS[op_key]['name']} ({op_key})")
        print(f"{'='*60}")

        data_file = f"{OPERATIONS[op_key]['data_dir']}/flatness_slt.csv"
        output_dir = get_figures_dir(op_key)

        if not os.path.exists(data_file):
            print(f"  跳过: 数据文件不存在 - {data_file}")
            continue

        print(f"  加载数据: {data_file}")
        data = load_flatness_data(data_file)
        print(f"  数据点数: {len(data['steps'])}")

        print(f"  生成图表...")
        os.makedirs(output_dir, exist_ok=True)

        plot_flatness_dual_axis(data, output_dir)


if __name__ == "__main__":
    main()
