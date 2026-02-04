#!/usr/bin/env python3
"""
统一 Gini/IPR 绘图 - 支持所有四种运算
将多子图分开保存，取消标题和序号

使用方法:
    python gini_ipr.py --operation x+y
    python gini_ipr.py --all
"""

import os
import sys
import csv
import argparse
import matplotlib.pyplot as plt
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
    COLORS = {
        'train_acc': '#3498db', 'test_acc': '#e74c3c',
        'gini': '#27ae60', 'ipr': '#8e44ad', 'grokking': '#f1c40f',
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


def load_gini_ipr_data(data_file):
    """加载 Gini/IPR 数据"""
    steps, train_accs, test_accs, ginis, iprs = [], [], [], [], []
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            train_accs.append(float(row['train_acc']))
            test_accs.append(float(row['test_acc']))
            ginis.append(float(row['gini']))
            iprs.append(float(row['ipr']))
    return steps, train_accs, test_accs, ginis, iprs


def plot_gini_coefficient(steps, ginis, train_accs, test_accs, output_dir, grokking_start=30000):
    """绘制 Gini 系数"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, ginis, color=COLORS['gini'], linewidth=2.5, label='Gini Coefficient')
    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Gini Coefficient', fontsize=FONTS['label']['size'], color=COLORS['gini'])
    ax.tick_params(labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTS['legend']['size'])

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'gini_coefficient')
    plt.close()


def plot_ipr(steps, iprs, output_dir, grokking_start=30000):
    """绘制 IPR"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, iprs, color=COLORS['ipr'], linewidth=2.5, label='IPR')
    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Inverse Participation Ratio', fontsize=FONTS['label']['size'], color=COLORS['ipr'])
    ax.tick_params(labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTS['legend']['size'])

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'inverse_participation_ratio')
    plt.close()


def plot_gini_with_accuracy(steps, ginis, train_accs, test_accs, output_dir, grokking_start=30000):
    """绘制 Gini 系数与准确率（双 y 轴）"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 左 y 轴：准确率
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

    # 右 y 轴：Gini 系数
    ax2 = ax.twinx()
    line3, = ax2.plot(steps, ginis, color=COLORS['gini'],
                      linewidth=2.5, alpha=0.9, label='Gini Coefficient')

    ax2.set_ylabel('Gini Coefficient', fontsize=FONTS['label']['size'], color=COLORS['gini'])
    ax2.tick_params(axis='y', labelcolor=COLORS['gini'], labelsize=FONTS['tick']['size'])

    # 合并图例
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='center left', fontsize=FONTS['legend']['size']-1)

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'gini_with_accuracy')
    plt.close()


def plot_ipr_with_accuracy(steps, iprs, train_accs, test_accs, output_dir, grokking_start=30000):
    """绘制 IPR 与准确率（双 y 轴）"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 左 y 轴：准确率
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

    # 右 y 轴：IPR
    ax2 = ax.twinx()
    line3, = ax2.plot(steps, iprs, color=COLORS['ipr'],
                      linewidth=2.5, alpha=0.9, label='IPR')

    ax2.set_ylabel('Inverse Participation Ratio', fontsize=FONTS['label']['size'], color=COLORS['ipr'])
    ax2.tick_params(axis='y', labelcolor=COLORS['ipr'], labelsize=FONTS['tick']['size'])

    # 合并图例
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='center left', fontsize=FONTS['legend']['size']-1)

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'ipr_with_accuracy')
    plt.close()


def plot_gini_change_rate(steps, ginis, output_dir, grokking_start=30000):
    """绘制 Gini 变化率"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ginis_diff = np.diff(ginis)
    ax.plot(steps[1:], ginis_diff, color=COLORS['gini'], linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Gini Change Rate', fontsize=FONTS['label']['size'])
    ax.tick_params(labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    add_grokking_region(ax, steps[1:], grokking_start)

    save_figure(fig, output_dir, 'gini_change_rate')
    plt.close()


def plot_ipr_change_rate(steps, iprs, output_dir, grokking_start=30000):
    """绘制 IPR 变化率"""
    fig, ax = plt.subplots(figsize=(10, 6))

    iprs_diff = np.diff(iprs)
    ax.plot(steps[1:], iprs_diff, color=COLORS['ipr'], linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('IPR Change Rate', fontsize=FONTS['label']['size'])
    ax.tick_params(labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    add_grokking_region(ax, steps[1:], grokking_start)

    save_figure(fig, output_dir, 'ipr_change_rate')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='绘制 Gini/IPR 图')
    parser.add_argument('--operation', type=str, default='x+y',
                        choices=['x+y', 'x-y', 'x*y', 'x_div_y', 'all'],
                        help='模运算类型')

    args = parser.parse_args()
    operations = [args.operation] if args.operation != 'all' else list(OPERATIONS.keys())

    for op_key in operations:
        print(f"\n{'='*60}")
        print(f"绘图: {OPERATIONS[op_key]['name']} ({op_key})")
        print(f"{'='*60}")

        data_file = f"{OPERATIONS[op_key]['data_dir']}/gini_ip_2d.csv"
        output_dir = get_figures_dir(op_key)

        if not os.path.exists(data_file):
            print(f"  跳过: 数据文件不存在 - {data_file}")
            continue

        print(f"  加载数据: {data_file}")
        steps, train_accs, test_accs, ginis, iprs = load_gini_ipr_data(data_file)
        print(f"  数据点数: {len(steps)}")

        print(f"  生成图表...")
        os.makedirs(output_dir, exist_ok=True)

        plot_gini_coefficient(steps, ginis, train_accs, test_accs, output_dir)
        print(f"    已保存: gini_coefficient")

        plot_ipr(steps, iprs, output_dir)
        print(f"    已保存: inverse_participation_ratio")

        plot_gini_with_accuracy(steps, ginis, train_accs, test_accs, output_dir)
        print(f"    已保存: gini_with_accuracy")

        plot_ipr_with_accuracy(steps, iprs, train_accs, test_accs, output_dir)
        print(f"    已保存: ipr_with_accuracy")

        plot_gini_change_rate(steps, ginis, output_dir)
        print(f"    已保存: gini_change_rate")

        plot_ipr_change_rate(steps, iprs, output_dir)
        print(f"    已保存: ipr_change_rate")


if __name__ == "__main__":
    main()
