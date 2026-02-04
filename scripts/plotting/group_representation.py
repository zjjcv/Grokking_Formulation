#!/usr/bin/env python3
"""
群论表示分析绘图 (Group Representation Plot) - 优化版本

绘制 Acc、相对残差 ε_R(t) 和正交性偏差 δ_orth(t) 随训练步数的变化

使用方法:
    python group_representation.py --operation x+y
    python group_representation.py --all
"""

import os
import sys
import csv
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# 配置 matplotlib
mpl.use('Agg')
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
    # 统一配色方案
    COLORS = {
        'train_acc': '#3498db',      # 蓝色
        'test_acc': '#e74c3c',       # 红色
        'epsilon_R': '#e67e22',       # 橙色 - 相对残差
        'delta_orth': '#9b59b6',     # 紫色 - 正交性偏差
        'grokking': '#f1c40f',       # 黄色
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


def load_group_representation_data(data_file):
    """加载群表示分析数据"""
    data = {
        'steps': [],
        'train_accs': [],
        'test_accs': [],
        'epsilon_R': [],
        'delta_orth': [],
        'closure_residual': [],
    }

    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['steps'].append(int(row['step']))
            data['train_accs'].append(float(row['train_acc']))
            data['test_accs'].append(float(row['test_acc']))
            data['epsilon_R'].append(float(row['epsilon_R']))
            data['delta_orth'].append(float(row['delta_orth']))
            if 'closure_residual' in row:
                data['closure_residual'].append(float(row['closure_residual']))

    return data


def plot_triple_axis(steps, train_accs, test_accs, epsilon_R, delta_orth, output_dir, grokking_start=30000):
    """绘制三 y 轴图：Acc, ε_R(t), δ_orth(t)"""
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # 左 y 轴：准确率（训练 + 测试）
    line_train, = ax1.plot(steps, train_accs, color=COLORS['train_acc'],
                           linewidth=1.5, alpha=0.5, label='Train Acc')
    line_test, = ax1.plot(steps, test_accs, color=COLORS['test_acc'],
                          linewidth=2.0, alpha=0.9, label='Test Acc')
    ax1.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax1.set_ylabel('Accuracy', fontsize=FONTS['label']['size'])
    ax1.tick_params(axis='y', labelsize=FONTS['tick']['size'])
    ax1.tick_params(axis='x', labelsize=FONTS['tick']['size'])
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # 右 y 轴 1：相对残差 ε_R(t)
    ax2 = ax1.twinx()
    line_epsilon, = ax2.plot(steps, epsilon_R, color=COLORS['epsilon_R'],
                             linewidth=2.5, alpha=0.9, label=r'$\epsilon_R(t)$')
    ax2.set_ylabel(r'Relative Residual $\epsilon_R(t)$',
                   fontsize=FONTS['label']['size'], color=COLORS['epsilon_R'], fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=COLORS['epsilon_R'], labelsize=FONTS['tick']['size'])

    # 右 y 轴 2：正交性偏差 δ_orth(t)
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('axes', 1.15))
    line_delta, = ax3.plot(steps, delta_orth, color=COLORS['delta_orth'],
                           linewidth=2.5, alpha=0.9, label=r'$\delta_{orth}(t)$')
    ax3.set_ylabel(r'Orthogonality $\delta_{orth}(t)$',
                   fontsize=FONTS['label']['size'], color=COLORS['delta_orth'], fontweight='bold')
    ax3.tick_params(axis='y', labelcolor=COLORS['delta_orth'], labelsize=FONTS['tick']['size'])

    # 合并图例
    lines = [line_train, line_test, line_epsilon, line_delta]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(1.22, 0.5),
              fontsize=FONTS['label']['size']-1, frameon=True, shadow=True)

    add_grokking_region(ax1, steps, grokking_start)

    plt.tight_layout()
    save_figure(fig, output_dir, 'group_representation_triple_axis')
    plt.close()


def plot_dual_residual(steps, train_accs, test_accs, epsilon_R, output_dir, grokking_start=30000):
    """绘制准确率 + 相对残差（双 y 轴）"""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左 y 轴：准确率
    line_train, = ax1.plot(steps, train_accs, color=COLORS['train_acc'],
                           linewidth=1.5, alpha=0.5, label='Train Acc')
    line_test, = ax1.plot(steps, test_accs, color=COLORS['test_acc'],
                          linewidth=2.0, alpha=0.9, label='Test Acc')
    ax1.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax1.set_ylabel('Accuracy', fontsize=FONTS['label']['size'])
    ax1.tick_params(axis='y', labelsize=FONTS['tick']['size'])
    ax1.tick_params(axis='x', labelsize=FONTS['tick']['size'])
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    ax1.legend(fontsize=FONTS['label']['size']-1, loc='best')

    # 右 y 轴：相对残差
    ax2 = ax1.twinx()
    line_epsilon, = ax2.plot(steps, epsilon_R, color=COLORS['epsilon_R'],
                             linewidth=2.5, alpha=0.9, label=r'$\epsilon_R(t)$')
    ax2.set_ylabel(r'Relative Residual $\epsilon_R(t)$',
                   fontsize=FONTS['label']['size'], color=COLORS['epsilon_R'], fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=COLORS['epsilon_R'], labelsize=FONTS['tick']['size'])
    ax2.legend([line_epsilon], [line_epsilon.get_label()], fontsize=FONTS['label']['size']-1, loc='center right')

    add_grokking_region(ax1, steps, grokking_start)

    save_figure(fig, output_dir, 'group_representation_residual')
    plt.close()


def plot_dual_orthogonality(steps, train_accs, test_accs, delta_orth, output_dir, grokking_start=30000):
    """绘制准确率 + 正交性偏差（双 y 轴）"""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左 y 轴：准确率
    line_train, = ax1.plot(steps, train_accs, color=COLORS['train_acc'],
                           linewidth=1.5, alpha=0.5, label='Train Acc')
    line_test, = ax1.plot(steps, test_accs, color=COLORS['test_acc'],
                          linewidth=2.0, alpha=0.9, label='Test Acc')
    ax1.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax1.set_ylabel('Accuracy', fontsize=FONTS['label']['size'])
    ax1.tick_params(axis='y', labelsize=FONTS['tick']['size'])
    ax1.tick_params(axis='x', labelsize=FONTS['tick']['size'])
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    ax1.legend(fontsize=FONTS['label']['size']-1, loc='best')

    # 右 y 轴：正交性偏差
    ax2 = ax1.twinx()
    line_delta, = ax2.plot(steps, delta_orth, color=COLORS['delta_orth'],
                           linewidth=2.5, alpha=0.9, label=r'$\delta_{orth}(t)$')
    ax2.set_ylabel(r'Orthogonality $\delta_{orth}(t)$',
                   fontsize=FONTS['label']['size'], color=COLORS['delta_orth'], fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=COLORS['delta_orth'], labelsize=FONTS['tick']['size'])
    ax2.legend([line_delta], [line_delta.get_label()], fontsize=FONTS['label']['size']-1, loc='center right')

    add_grokking_region(ax1, steps, grokking_start)

    save_figure(fig, output_dir, 'group_representation_orthogonality')
    plt.close()


def plot_closure_residual(steps, test_accs, closure_residual, output_dir, grokking_start=30000):
    """绘制环闭合残差"""
    if not closure_residual or all(v == 0 for v in closure_residual):
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左 y 轴：准确率
    line_test, = ax1.plot(steps, test_accs, color=COLORS['test_acc'],
                          linewidth=2.0, alpha=0.9, label='Test Acc')
    ax1.set_xlabel('Training Step', fontsize=FONTS['label']['size'])
    ax1.set_ylabel('Accuracy', fontsize=FONTS['label']['size'], color=COLORS['test_acc'])
    ax1.tick_params(axis='y', labelcolor=COLORS['test_acc'], labelsize=FONTS['tick']['size'])
    ax1.tick_params(axis='x', labelsize=FONTS['tick']['size'])
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    ax1.legend(fontsize=FONTS['label']['size']-1, loc='best')

    # 右 y 轴：环闭合残差
    ax2 = ax1.twinx()
    line_closure, = ax2.plot(steps, closure_residual, color='#16a085',
                             linewidth=2.5, alpha=0.9, label='Closure Residual')
    ax2.set_ylabel('Closure Residual', fontsize=FONTS['label']['size'],
                   color='#16a085', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#16a085', labelsize=FONTS['tick']['size'])
    ax2.legend([line_closure], [line_closure.get_label()], fontsize=FONTS['label']['size']-1, loc='center right')

    add_grokking_region(ax1, steps, grokking_start)

    save_figure(fig, output_dir, 'group_representation_closure')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='绘制群论表示分析图')
    parser.add_argument('--operation', type=str, default='x+y',
                        choices=['x+y', 'x-y', 'x*y', 'x_div_y', 'all'],
                        help='模运算类型')

    args = parser.parse_args()

    operations = [args.operation] if args.operation != 'all' else list(OPERATIONS.keys())

    for op_key in operations:
        print(f"\n{'='*60}")
        print(f"绘图: {OPERATIONS[op_key]['name']} ({op_key})")
        print(f"{'='*60}")

        data_file = f"{OPERATIONS[op_key]['data_dir']}/group_representation.csv"
        output_dir = get_figures_dir(op_key)

        if not os.path.exists(data_file):
            print(f"  跳过: 数据文件不存在 - {data_file}")
            continue

        print(f"  加载数据: {data_file}")
        data = load_group_representation_data(data_file)
        print(f"  数据点数: {len(data['steps'])}")

        print(f"  生成图表...")
        os.makedirs(output_dir, exist_ok=True)

        # 三 y 轴图
        plot_triple_axis(data['steps'], data['train_accs'], data['test_accs'],
                        data['epsilon_R'], data['delta_orth'], output_dir)
        print(f"    已保存: group_representation_triple_axis")

        # 双 y 轴图
        plot_dual_residual(data['steps'], data['train_accs'], data['test_accs'],
                          data['epsilon_R'], output_dir)
        print(f"    已保存: group_representation_residual")

        plot_dual_orthogonality(data['steps'], data['train_accs'], data['test_accs'],
                                data['delta_orth'], output_dir)
        print(f"    已保存: group_representation_orthogonality")

        # 环闭合残差
        if data['closure_residual']:
            plot_closure_residual(data['steps'], data['test_accs'],
                                 data['closure_residual'], output_dir)
            print(f"    已保存: group_representation_closure")


if __name__ == "__main__":
    main()
