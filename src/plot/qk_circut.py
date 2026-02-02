#!/usr/bin/env python3
"""
绘制 QK 电路的二维频谱图

使用 qk_circut.csv 数据，为每个步数绘制：
1. 频谱幅度热力图（2D DFT magnitude）
2. 对数刻度的频谱图
3. 显示关键统计信息
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors


def load_qk_circuit_data(qk_circut_file):
    """加载 QK 电路数据"""
    data = []
    with open(qk_circut_file, 'r') as f:
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


def plot_qk_circuit_2d_spectrum(qk_data, p, output_dir):
    """绘制 QK 电路二维频谱图"""
    steps = [int(row['step']) for row in qk_data]

    # 创建图形 - 每个步数一个子图
    n_steps = len(steps)
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

    # 为每个步数绘制频谱图
    for idx, row in enumerate(qk_data):
        step = int(row['step'])
        test_acc = float(row['test_acc'])

        # 重建频谱
        magnitude = reconstruct_magnitude_spectrum(row, p)

        # 计算对数频谱（加小常数避免 log(0)）
        log_magnitude = np.log10(magnitude + 1e-10)

        # 选择子图位置
        if idx < 4:
            ax = fig.add_subplot(gs[0, idx])
        else:
            ax = fig.add_subplot(gs[1, idx - 4])

        # 绘制线性刻度频谱
        im = ax.imshow(magnitude, cmap='viridis', origin='lower',
                       interpolation='bilinear', aspect='equal')

        # 设置标题和标签
        ax.set_title(f'Step {step:,} | Test Acc: {test_acc:.1%}\n'
                     f'DC: {float(row["dc_component"]):.2e} | '
                     f'Entropy: {float(row["spectral_entropy"]):.2f}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Frequency k', fontsize=9)
        ax.set_ylabel('Frequency l', fontsize=9)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Magnitude', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    # 隐藏最后一个空子图（如果有）
    if n_steps < 8:
        ax = fig.add_subplot(gs[1, 3])
        ax.axis('off')

    # 设置整体标题
    fig.suptitle('QK Circuit 2D Frequency Spectrum Analysis\n'
                 'A = W_E @ W_Q @ W_K^T @ W_E^T (p x p matrix)',
                 fontsize=16, fontweight='bold', y=0.98)

    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'qk_circut_2d_spectrum.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file}')

    # 同时保存 PDF 版本
    output_file_pdf = os.path.join(output_dir, 'qk_circut_2d_spectrum.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file_pdf}')

    plt.close()


def plot_qk_circuit_log_spectrum(qk_data, p, output_dir):
    """绘制 QK 电路对数频谱图"""
    steps = [int(row['step']) for row in qk_data]

    # 创建图形 - 每个步数一个子图
    n_steps = len(steps)
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

    # 为每个步数绘制对数频谱图
    for idx, row in enumerate(qk_data):
        step = int(row['step'])
        test_acc = float(row['test_acc'])

        # 重建频谱
        magnitude = reconstruct_magnitude_spectrum(row, p)

        # 计算对数频谱（dB刻度）
        log_magnitude = 20 * np.log10(magnitude / np.max(magnitude) + 1e-10)

        # 选择子图位置
        if idx < 4:
            ax = fig.add_subplot(gs[0, idx])
        else:
            ax = fig.add_subplot(gs[1, idx - 4])

        # 绘制对数刻度频谱
        im = ax.imshow(log_magnitude, cmap='RdBu_r', origin='lower',
                       interpolation='bilinear', aspect='equal',
                       vmin=-60, vmax=0)

        # 设置标题和标签
        ax.set_title(f'Step {step:,} | Test Acc: {test_acc:.1%}\n'
                     f'LowFreq: {float(row["low_freq_energy"]):.2e} | '
                     f'Spectral Entropy: {float(row["spectral_entropy"]):.2f}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Frequency k', fontsize=9)
        ax.set_ylabel('Frequency l', fontsize=9)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Magnitude (dB)', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    # 隐藏最后一个空子图（如果有）
    if n_steps < 8:
        ax = fig.add_subplot(gs[1, 3])
        ax.axis('off')

    # 设置整体标题
    fig.suptitle('QK Circuit 2D Frequency Spectrum (Log Scale / dB)\n'
                 'A = W_E @ W_Q @ W_K^T @ W_E^T (p x p matrix)',
                 fontsize=16, fontweight='bold', y=0.98)

    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'qk_circut_2d_spectrum_log.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file}')

    # 同时保存 PDF 版本
    output_file_pdf = os.path.join(output_dir, 'qk_circut_2d_spectrum_log.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file_pdf}')

    plt.close()


def plot_qk_circuit_evolution(qk_data, output_dir):
    """绘制频谱统计量随训练步数的变化"""
    steps = [int(row['step']) for row in qk_data]
    test_accs = [float(row['test_acc']) for row in qk_data]
    dc_components = [float(row['dc_component']) for row in qk_data]
    low_freq_energies = [float(row['low_freq_energy']) for row in qk_data]
    spectral_entropies = [float(row['spectral_entropy']) for row in qk_data]

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 测试准确率
    ax = axes[0, 0]
    ax.plot(steps, test_accs, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Test Accuracy', fontsize=11)
    ax.set_title('Test Accuracy Evolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # 2. DC 分量
    ax = axes[0, 1]
    ax.plot(steps, dc_components, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('DC Component (Magnitude at (0,0))', fontsize=11)
    ax.set_title('DC Component Evolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # 3. 低频能量
    ax = axes[1, 0]
    ax.plot(steps, low_freq_energies, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Low Frequency Energy (5x5 region)', fontsize=11)
    ax.set_title('Low Frequency Energy Evolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # 4. 频谱熵
    ax = axes[1, 1]
    ax.plot(steps, spectral_entropies, 'mo-', linewidth=2, markersize=8)
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Spectral Entropy', fontsize=11)
    ax.set_title('Spectral Entropy Evolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.suptitle('QK Circuit Frequency Statistics Evolution',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'qk_circut_evolution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file}')

    # 同时保存 PDF 版本
    output_file_pdf = os.path.join(output_dir, 'qk_circut_evolution.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file_pdf}')

    plt.close()


def main():
    """主函数"""
    # 配置参数
    qk_circut_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/qk_circut.csv"
    output_dir = "/root/data1/zjj/Grokking_Formulation/experiments/figures"
    p = 97  # 模数

    print("=" * 60)
    print("绘制 QK 电路二维频谱图")
    print("=" * 60)
    print(f"QK 电路数据文件: {qk_circut_file}")
    print(f"输出目录: {output_dir}")
    print(f"模数 p: {p}")
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    qk_data = load_qk_circuit_data(qk_circut_file)
    print(f"数据点数: {len(qk_data)}")

    # 打印数据摘要
    print("\n数据摘要:")
    for row in qk_data:
        print(f"  Step {row['step']}: Test Acc={float(row['test_acc']):.1%}, "
              f"DC={float(row['dc_component']):.2e}, "
              f"Entropy={float(row['spectral_entropy']):.2f}")

    # 绘制图形
    print("\n生成线性刻度频谱图...")
    plot_qk_circuit_2d_spectrum(qk_data, p, output_dir)

    print("\n生成对数刻度频谱图...")
    plot_qk_circuit_log_spectrum(qk_data, p, output_dir)

    print("\n生成频谱统计量演化图...")
    plot_qk_circuit_evolution(qk_data, output_dir)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
