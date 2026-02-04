#!/usr/bin/env python3
"""
统一 QK 电路 2D 频谱热图绘图 - 支持所有四种运算

使用完整的频谱数据绘制 2D 热图

使用方法:
    python qk_spectrum.py --operation x+y
    python qk_spectrum.py --all
"""

import os
import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 添加父目录到路径
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in __file__:
    sys.path.insert(0, sys_path)

try:
    from lib.config import (
        OPERATIONS, COLORS, FONTS,
        get_figures_dir, save_figure
    )
except ImportError:
    COLORS = {
        'train_acc': '#3498db', 'test_acc': '#e74c3c',
        'grokking': '#f1c40f',
    }
    FONTS = {'label': {'size': 12}, 'tick': {'size': 10}, 'legend': {'size': 10}}
    OPERATIONS = {
        'x+y': {'name': 'Addition', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x+y'},
        'x-y': {'name': 'Subtraction', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x-y'},
        'x*y': {'name': 'Multiplication', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x*y'},
        'x_div_y': {'name': 'Division', 'data_dir': '/root/data1/zjj/Grokking_Formulation/data/x_div_y'},
    }
    def get_figures_dir(op): return f"/root/data1/zjj/Grokking_Formulation/experiments/figures/{op}"
    def save_figure(fig, output_dir, filename):
        os.makedirs(output_dir, exist_ok=True)
        for fmt in ['png', 'pdf']:
            filepath = os.path.join(output_dir, f'{filename}.{fmt}')
            fig.savefig(filepath, bbox_inches='tight', dpi=300 if fmt == 'png' else None)


def load_qk_spectrum_data(data_file):
    """加载 QK 电路频谱数据"""
    data = []
    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def reconstruct_magnitude_spectrum(row, p):
    """从 CSV 行重建二维频谱"""
    magnitude = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            magnitude[i, j] = float(row[f'mag_{i}_{j}'])
    return magnitude


def plot_qk_2d_spectrum(qk_data, p, output_dir, op_name):
    """绘制 QK 电路二维频谱图（线性刻度）"""
    n_steps = len(qk_data)

    # 根据数据点数确定布局
    if n_steps <= 4:
        nrows, ncols = 1, n_steps
        figsize = (5 * n_steps, 5)
    elif n_steps <= 8:
        nrows, ncols = 2, 4
        figsize = (18, 10)
    else:
        nrows, ncols = 3, 4
        figsize = (18, 14)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.3, wspace=0.3)

    for idx, row in enumerate(qk_data):
        step = int(row['step'])
        test_acc = float(row['test_acc'])

        # 重建频谱
        magnitude = reconstruct_magnitude_spectrum(row, p)

        # 选择子图位置
        ax = fig.add_subplot(gs[idx])

        # 绘制线性刻度频谱
        im = ax.imshow(magnitude, cmap='viridis', origin='lower',
                       interpolation='bilinear', aspect='equal')

        # 设置标题和标签
        ax.set_title(f'Step {step:,} | Test Acc: {test_acc:.1%}\n'
                     f'DC: {float(row["dc_component"]):.2e} | '
                     f'Entropy: {float(row["spectral_entropy"]):.2f}',
                     fontsize=FONTS['label']['size'], fontweight='bold')
        ax.set_xlabel('Frequency k', fontsize=FONTS['tick']['size'])
        ax.set_ylabel('Frequency l', fontsize=FONTS['tick']['size'])
        ax.tick_params(labelsize=FONTS['tick']['size']-1)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Magnitude', fontsize=FONTS['tick']['size'])
        cbar.ax.tick_params(labelsize=FONTS['tick']['size']-1)

    # 隐藏多余的子图
    for idx in range(n_steps, nrows * ncols):
        ax = fig.add_subplot(gs[idx])
        ax.axis('off')

    save_figure(fig, output_dir, 'qk_circut_2d_spectrum')
    plt.close()


def plot_qk_2d_spectrum_log(qk_data, p, output_dir, op_name):
    """绘制 QK 电路二维频谱图（对数刻度）"""
    n_steps = len(qk_data)

    # 根据数据点数确定布局
    if n_steps <= 4:
        nrows, ncols = 1, n_steps
        figsize = (5 * n_steps, 5)
    elif n_steps <= 8:
        nrows, ncols = 2, 4
        figsize = (18, 10)
    else:
        nrows, ncols = 3, 4
        figsize = (18, 14)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.3, wspace=0.3)

    for idx, row in enumerate(qk_data):
        step = int(row['step'])
        test_acc = float(row['test_acc'])

        # 重建频谱
        magnitude = reconstruct_magnitude_spectrum(row, p)

        # 计算对数频谱（dB刻度）
        max_val = np.max(magnitude)
        if max_val > 0:
            log_magnitude = 20 * np.log10(magnitude / max_val + 1e-10)
        else:
            log_magnitude = magnitude

        # 选择子图位置
        ax = fig.add_subplot(gs[idx])

        # 绘制对数刻度频谱
        im = ax.imshow(log_magnitude, cmap='RdBu_r', origin='lower',
                       interpolation='bilinear', aspect='equal',
                       vmin=-60, vmax=0)

        # 设置标题和标签（两行，减小字体）
        ax.set_title(f'Step {step:,} | Test Acc: {test_acc:.1%}\n'
                     f'LowFreq: {float(row["low_freq_energy"]):.2e} | Entropy: {float(row["spectral_entropy"]):.2f}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Frequency k', fontsize=FONTS['tick']['size'])
        ax.set_ylabel('Frequency l', fontsize=FONTS['tick']['size'])
        ax.tick_params(labelsize=FONTS['tick']['size']-1)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Magnitude (dB)', fontsize=FONTS['tick']['size'])
        cbar.ax.tick_params(labelsize=FONTS['tick']['size']-1)

    # 隐藏多余的子图
    for idx in range(n_steps, nrows * ncols):
        ax = fig.add_subplot(gs[idx])
        ax.axis('off')

    save_figure(fig, output_dir, 'qk_circut_2d_spectrum_log')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='绘制 QK 电路 2D 频谱热图')
    parser.add_argument('--operation', type=str, default='x+y',
                        choices=['x+y', 'x-y', 'x*y', 'x_div_y', 'all'],
                        help='模运算类型')
    parser.add_argument('--p', type=int, default=97, help='模数')

    args = parser.parse_args()

    operations = [args.operation] if args.operation != 'all' else list(OPERATIONS.keys())

    for op_key in operations:
        print(f"\n{'='*60}")
        print(f"绘图: {OPERATIONS[op_key]['name']} ({op_key})")
        print(f"{'='*60}")

        data_file = f"{OPERATIONS[op_key]['data_dir']}/qk_circut_full.csv"
        output_dir = get_figures_dir(op_key)

        if not os.path.exists(data_file):
            print(f"  跳过: 数据文件不存在 - {data_file}")
            print(f"  请先运行: python src/2D_DFT_unified/qk_circut_full.py --operation {op_key}")
            continue

        print(f"  加载数据: {data_file}")
        qk_data = load_qk_spectrum_data(data_file)
        print(f"  数据点数: {len(qk_data)}")

        print(f"  生成图表...")
        os.makedirs(output_dir, exist_ok=True)

        plot_qk_2d_spectrum(qk_data, args.p, output_dir, OPERATIONS[op_key]['name'])
        print(f"    已保存: qk_circut_2d_spectrum")

        plot_qk_2d_spectrum_log(qk_data, args.p, output_dir, OPERATIONS[op_key]['name'])
        print(f"    已保存: qk_circut_2d_spectrum_log")


if __name__ == "__main__":
    main()
