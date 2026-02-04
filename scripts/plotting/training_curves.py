#!/usr/bin/env python3
"""
统一训练曲线绘图 - 支持所有四种运算
子图单独保存，取消标题和序号，统一配色

使用方法:
    python training_curves.py --operation x+y
    python training_curves.py --all
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
        OPERATIONS, COLORS, FONTS, LINES,
        get_metric_file, get_figures_dir,
        create_figure, set_log_scale, add_grokking_region, save_figure
    )
except ImportError:
    # 默认配置
    COLORS = {
        'train_acc': '#3498db', 'test_acc': '#e74c3c',
        'train_loss': '#95a5a6', 'test_loss': '#7f8c8d',
        'grokking': '#f1c40f',
    }
    FONTS = {'label': {'size': 12}, 'tick': {'size': 10}}
    LINES = {
        'train_acc': {'linewidth': 1.5, 'alpha': 0.7},
        'test_acc': {'linewidth': 2.0, 'alpha': 0.9},
    }
    OPERATIONS = {
        'x+y': {'name': 'Addition', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x+y'},
        'x-y': {'name': 'Subtraction', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x-y'},
        'x*y': {'name': 'Multiplication', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x*y'},
        'x_div_y': {'name': 'Division', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x_div_y'},
    }
    def get_metric_file(op): return f"{OPERATIONS[op]['data_dir']}/metric.csv"
    def get_figures_dir(op): return f"/root/data1/zjj/Grokking_Formulation/experiments/figures/{op}"
    def add_grokking_region(ax, steps, start_step=30000):
        max_step = max(steps) if hasattr(steps, '__iter__') else steps
        ax.axvspan(start_step, max_step, alpha=0.1, color=COLORS['grokking'])
    def save_figure(fig, output_dir, filename):
        os.makedirs(output_dir, exist_ok=True)
        for fmt in ['png', 'pdf']:
            filepath = os.path.join(output_dir, f'{filename}.{fmt}')
            fig.savefig(filepath, bbox_inches='tight', dpi=300 if fmt == 'png' else None)


def load_metric_data(metric_file):
    """加载训练指标数据"""
    steps, train_losses, train_accs, test_losses, test_accs = [], [], [], [], []
    with open(metric_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            train_losses.append(float(row['train_loss']))
            train_accs.append(float(row['train_acc']))
            test_losses.append(float(row['test_loss']))
            test_accs.append(float(row['test_acc']))
    return steps, train_losses, train_accs, test_losses, test_accs


def plot_accuracy(steps, train_accs, test_accs, output_dir, grokking_start=30000):
    """绘制准确率曲线（单图）"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, train_accs, color=COLORS['train_acc'],
            linewidth=LINES['train_acc']['linewidth'],
            alpha=LINES['train_acc']['alpha'], label='Train Accuracy')
    ax.plot(steps, test_accs, color=COLORS['test_acc'],
            linewidth=LINES['test_acc']['linewidth'],
            alpha=LINES['test_acc']['alpha'], label='Test Accuracy')

    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Accuracy', fontsize=FONTS['label']['size'])
    ax.tick_params(labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTS['label']['size'])
    ax.set_ylim([0, 1.05])

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'training_accuracy')
    plt.close()


def plot_loss(steps, train_losses, test_losses, output_dir, grokking_start=30000):
    """绘制损失曲线（单图）"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, train_losses, color=COLORS['train_loss'],
            linewidth=LINES['train_acc']['linewidth'],
            alpha=LINES['train_acc']['alpha'], label='Train Loss')
    ax.plot(steps, test_losses, color=COLORS['test_loss'],
            linewidth=LINES['test_acc']['linewidth'],
            alpha=LINES['test_acc']['alpha'], label='Test Loss')

    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Loss', fontsize=FONTS['label']['size'])
    ax.tick_params(labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONTS['label']['size'])

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'training_loss')
    plt.close()


def plot_combined(steps, train_losses, train_accs, test_losses, test_accs, output_dir, grokking_start=30000):
    """绘制组合图（准确率+损失，双 y 轴）"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 左 y 轴：准确率
    line1, = ax.plot(steps, train_accs, color=COLORS['train_acc'],
                     linewidth=1.5, alpha=0.7, label='Train Acc')
    line2, = ax.plot(steps, test_accs, color=COLORS['test_acc'],
                     linewidth=2.0, alpha=0.9, label='Test Acc')

    ax.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax.set_ylabel('Accuracy', fontsize=FONTS['label']['size'], color=COLORS['train_acc'])
    ax.tick_params(axis='y', labelcolor=COLORS['train_acc'], labelsize=FONTS['tick']['size'])
    ax.tick_params(axis='x', labelsize=FONTS['tick']['size'])
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # 右 y 轴：损失
    ax2 = ax.twinx()
    line3, = ax2.plot(steps, train_losses, color=COLORS['train_loss'],
                      linewidth=1.5, alpha=0.5, linestyle='--', label='Train Loss')
    line4, = ax2.plot(steps, test_losses, color=COLORS['test_loss'],
                      linewidth=1.5, alpha=0.5, linestyle=':', label='Test Loss')

    ax2.set_ylabel('Loss', fontsize=FONTS['label']['size'], color=COLORS['train_loss'])
    ax2.tick_params(axis='y', labelcolor=COLORS['train_loss'], labelsize=FONTS['tick']['size'])

    # 合并图例
    lines = [line1, line2, line3, line4]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='center left', fontsize=FONTS['label']['size']-1)

    add_grokking_region(ax, steps, grokking_start)

    save_figure(fig, output_dir, 'training_curves_combined')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='绘制训练曲线')
    parser.add_argument('--operation', type=str, default='x+y',
                        choices=['x+y', 'x-y', 'x*y', 'x_div_y', 'all'],
                        help='模运算类型')

    args = parser.parse_args()
    operations = [args.operation] if args.operation != 'all' else list(OPERATIONS.keys())

    for op_key in operations:
        print(f"\n{'='*60}")
        print(f"绘图: {OPERATIONS[op_key]['name']} ({op_key})")
        print(f"{'='*60}")

        metric_file = get_metric_file(op_key)
        output_dir = get_figures_dir(op_key)

        if not os.path.exists(metric_file):
            print(f"  跳过: 数据文件不存在 - {metric_file}")
            continue

        print(f"  加载数据: {metric_file}")
        steps, train_losses, train_accs, test_losses, test_accs = load_metric_data(metric_file)
        print(f"  数据点数: {len(steps)}")

        print(f"  生成图表...")
        os.makedirs(output_dir, exist_ok=True)

        plot_accuracy(steps, train_accs, test_accs, output_dir)
        print(f"    已保存: training_accuracy")

        plot_loss(steps, train_losses, test_losses, output_dir)
        print(f"    已保存: training_loss")

        plot_combined(steps, train_losses, train_accs, test_losses, test_accs, output_dir)
        print(f"    已保存: training_curves_combined")


if __name__ == "__main__":
    main()
