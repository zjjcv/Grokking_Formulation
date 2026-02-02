#!/usr/bin/env python3
"""
绘制 2D 功率谱密度随训练步数的变化

使用 psd_2d.csv 和 metric.csv 绘制双 y 轴图：
- 左 y 轴：训练和测试准确率
- 右 y 轴：PSD 统计量（DC 分量、低频能量、频谱熵）
- x 轴采用对数刻度
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_data(psd_file, metric_file):
    """加载 PSD 和 metric 数据"""
    psd_data = []
    with open(psd_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            psd_data.append(row)

    metric_data = []
    with open(metric_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_data.append(row)

    return psd_data, metric_data


def plot_2d_psd(psd_data, metric_data, output_dir):
    """绘制 2D PSD 图"""
    steps = [int(row['step']) for row in psd_data]
    train_accs = [float(row['train_acc']) for row in psd_data]
    test_accs = [float(row['test_acc']) for row in psd_data]
    dc_components = [float(row['dc_component']) for row in psd_data]
    low_freq_energies = [float(row['low_freq_energy']) for row in psd_data]
    spectral_entropies = [float(row['spectral_entropy']) for row in psd_data]

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 子图 1: DC 分量
    ax = axes[0, 0]
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    line1, = ax.plot(steps, train_accs, 'b-', label='Train Acc', linewidth=1.5, alpha=0.5)
    line2, = ax.plot(steps, test_accs, 'r-', label='Test Acc', linewidth=2, alpha=0.7)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('DC Component (2D DFT)', fontsize=12, fontweight='bold')

    ax2 = ax.twinx()
    ax2.set_ylabel('DC Component', fontsize=11, color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    line3, = ax2.plot(steps, dc_components, 'orange', linewidth=2, alpha=0.8)
    ax2.legend([line3], [line3.get_label()], loc='upper right', fontsize=9)

    # 子图 2: 低频能量
    ax = axes[0, 1]
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    line1, = ax.plot(steps, train_accs, 'b-', label='Train Acc', linewidth=1.5, alpha=0.5)
    line2, = ax.plot(steps, test_accs, 'r-', label='Test Acc', linewidth=2, alpha=0.7)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('Low Frequency Energy (2D DFT)', fontsize=12, fontweight='bold')

    ax2 = ax.twinx()
    ax2.set_ylabel('Low Freq Energy', fontsize=11, color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    line3, = ax2.plot(steps, low_freq_energies, 'green', linewidth=2, alpha=0.8)
    ax2.legend([line3], [line3.get_label()], loc='upper right', fontsize=9)

    # 子图 3: 频谱熵
    ax = axes[1, 0]
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    line1, = ax.plot(steps, train_accs, 'b-', label='Train Acc', linewidth=1.5, alpha=0.5)
    line2, = ax.plot(steps, test_accs, 'r-', label='Test Acc', linewidth=2, alpha=0.7)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('Spectral Entropy (2D DFT)', fontsize=12, fontweight='bold')

    ax2 = ax.twinx()
    ax2.set_ylabel('Spectral Entropy', fontsize=11, color='tab:purple')
    ax2.tick_params(axis='y', labelcolor='tab:purple')
    line3, = ax2.plot(steps, spectral_entropies, 'purple', linewidth=2, alpha=0.8)
    ax2.legend([line3], [line3.get_label()], loc='upper right', fontsize=9)

    # 子图 4: 综合对比
    ax = axes[1, 1]
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    line1, = ax.plot(steps, train_accs, 'b-', label='Train Acc', linewidth=1.5, alpha=0.5)
    line2, = ax.plot(steps, test_accs, 'r-', label='Test Acc', linewidth=2, alpha=0.7)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('2D PSD: Combined Metrics', fontsize=12, fontweight='bold')

    ax2 = ax.twinx()
    ax2.set_ylabel('Normalized Value', fontsize=11)

    # 归一化所有指标到 [0, 1]
    dc_norm = (np.array(dc_components) - np.min(dc_components)) / (np.max(dc_components) - np.min(dc_components))
    low_freq_norm = (np.array(low_freq_energies) - np.min(low_freq_energies)) / (np.max(low_freq_energies) - np.min(low_freq_energies))

    ax2.plot(steps, dc_norm, 'orange', linewidth=1.5, alpha=0.7, label='DC (norm)')
    ax2.plot(steps, low_freq_norm, 'green', linewidth=1.5, alpha=0.7, label='LowFreq (norm)')
    ax2.legend(loc='upper right', fontsize=9)

    # 添加 Grokking 区域标注
    for ax in axes.flat:
        grokking_start = 30000
        ax.axvspan(grokking_start, max(steps), alpha=0.1, color='yellow')

    plt.suptitle('Grokking: 2D Power Spectral Density Analysis\n'
                 'DC: (0,0) frequency | LowFreq: 5x5 region | Entropy: frequency distribution',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'psd_2d_metrics.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file}')

    output_file_pdf = os.path.join(output_dir, 'psd_2d_metrics.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file_pdf}')

    plt.close()


def main():
    psd_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/psd_2d.csv"
    metric_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/metric.csv"
    output_dir = "/root/data1/zjj/Grokking_Formulation/experiments/figures/2D_DFT"

    print("=" * 60)
    print("绘制 2D 功率谱密度变化图")
    print("=" * 60)
    print(f"PSD 数据文件: {psd_file}")
    print(f"Metric 数据文件: {metric_file}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    print("\n加载数据...")
    psd_data, metric_data = load_data(psd_file, metric_file)
    print(f"PSD 数据点数: {len(psd_data)}")
    print(f"Metric 数据点数: {len(metric_data)}")

    print("\n生成图形...")
    plot_2d_psd(psd_data, metric_data, output_dir)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
