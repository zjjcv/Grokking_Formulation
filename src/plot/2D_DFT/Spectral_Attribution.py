#!/usr/bin/env python3
"""
绘制频谱归因分析结果

使用 spectral_attribution.csv 绘制：
1. 热力图：训练步数 × 频率，颜色表示贡献值
2. 堆叠面积图：低频 vs 高频贡献演化
3. 频率带演化图
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_spectral_attribution_data(spectral_file):
    """加载频谱归因数据"""
    data = []
    with open(spectral_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def plot_spectral_heatmap(data, output_dir):
    """绘制频谱归因热力图"""
    steps = [int(row['step']) for row in data]
    test_accs = [float(row['test_acc']) for row in data]

    # 提取频率归因数据
    max_freq = 20
    freq_columns = [f'freq_{k}_attribution' for k in range(max_freq)]

    # 构建矩阵 (n_steps, n_freqs)
    attribution_matrix = []
    for row in data:
        freq_values = [float(row.get(col, 0)) for col in freq_columns]
        attribution_matrix.append(freq_values)

    attribution_matrix = np.array(attribution_matrix)

    # 创建图形
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3, width_ratios=[3, 1], height_ratios=[1, 1])

    # 子图 1: 热力图（大图）
    ax_heatmap = fig.add_subplot(gs[:, 0])

    # 对数刻度的 x 轴
    extent = [0, max_freq, min(steps), max(steps)]

    im = ax_heatmap.imshow(
        attribution_matrix,
        aspect='auto',
        origin='lower',
        cmap='RdYlBu_r',
        extent=[0, max_freq, min(steps), max(steps)],
        interpolation='nearest'
    )

    ax_heatmap.set_xscale('linear')
    ax_heatmap.set_yscale('log')
    ax_heatmap.set_xlabel('Frequency k', fontsize=12, fontweight='bold')
    ax_heatmap.set_ylabel('Training Step (log scale)', fontsize=12, fontweight='bold')
    ax_heatmap.set_title('Spectral Attribution Heatmap\\nFrequency Contribution to Logit',
                        fontsize=14, fontweight='bold')

    # 添加 Grokking 区域标注
    ax_heatmap.axhspan(30000, max(steps), alpha=0.1, color='yellow')

    # 颜色条
    cbar = plt.colorbar(im, ax=ax_heatmap, pad=0.02)
    cbar.set_label('Attribution Value (Energy)', fontsize=11)

    # 标注关键频率
    ax_heatmap.axvline(x=5.5, color='white', linestyle='--', alpha=0.7, linewidth=2)
    ax_heatmap.text(2.5, max(steps) * 0.95, 'Low Freq', color='white',
                   fontsize=11, fontweight='bold', ha='center')
    ax_heatmap.text(12.5, max(steps) * 0.95, 'High Freq', color='white',
                   fontsize=11, fontweight='bold', ha='center')

    # 子图 2: 测试准确率曲线
    ax_acc = fig.add_subplot(gs[0, 1])
    ax_acc.plot(steps, test_accs, 'b-', linewidth=2)
    ax_acc.axvline(x=30000, color='gray', linestyle='--', alpha=0.5, label='Grokking Start')
    ax_acc.set_xscale('log')
    ax_acc.set_xlabel('Training Step', fontsize=10)
    ax_acc.set_ylabel('Test Accuracy', fontsize=10)
    ax_acc.set_title('Test Accuracy', fontsize=11, fontweight='bold')
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(fontsize=9)

    # 子图 3: 低频 vs 高频比值
    ax_ratio = fig.add_subplot(gs[1, 1])
    ratios = [float(row.get('low_high_ratio', 0)) for row in data]
    ax_ratio.plot(steps, ratios, 'purple', linewidth=2)
    ax_ratio.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax_ratio.axvline(x=30000, color='gray', linestyle='--', alpha=0.5)
    ax_ratio.set_xscale('log')
    ax_ratio.set_xlabel('Training Step', fontsize=10)
    ax_ratio.set_ylabel('Low/High Freq Ratio', fontsize=10)
    ax_ratio.set_title('Frequency Dominance Ratio', fontsize=11, fontweight='bold')
    ax_ratio.grid(True, alpha=0.3)

    plt.suptitle('Grokking: Spectral Attribution Analysis\\n'
                'How different frequency components contribute to predictions over training',
                fontsize=15, fontweight='bold', y=0.98)

    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'spectral_attribution_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file}')

    output_file_pdf = os.path.join(output_dir, 'spectral_attribution_heatmap.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file_pdf}')

    plt.close()


def plot_frequency_band_evolution(data, output_dir):
    """绘制频率带演化图"""
    steps = [int(row['step']) for row in data]
    test_accs = [float(row['test_acc']) for row in data]
    low_freq = [float(row.get('low_freq_attribution', 0)) for row in data]
    high_freq = [float(row.get('high_freq_attribution', 0)) for row in data]

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 子图 1: 堆叠面积图
    ax = axes[0, 0]
    ax.fill_between(steps, 0, low_freq, alpha=0.6, color='blue', label='Low Freq (k≤5)')
    ax.fill_between(steps, low_freq,
                    np.array(low_freq) + np.array(high_freq),
                    alpha=0.6, color='red', label='High Freq (k>5)')
    ax.set_xscale('log')
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Total Attribution', fontsize=11)
    ax.set_title('Frequency Band Contribution (Stacked)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    # 子图 2: 低频 vs 高频（重叠）
    ax = axes[0, 1]
    ax.plot(steps, low_freq, 'b-', linewidth=2, label='Low Freq', alpha=0.7)
    ax.plot(steps, high_freq, 'r-', linewidth=2, label='High Freq', alpha=0.7)
    ax.axvline(x=30000, color='gray', linestyle='--', alpha=0.5, label='Grokking Start')
    ax.set_xscale('log')
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Attribution', fontsize=11)
    ax.set_title('Low vs High Frequency (Overlaid)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 子图 3: 比值 + 准确率（双 y 轴）
    ax = axes[1, 0]
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Low/High Ratio', fontsize=11, color='purple')
    ax.tick_params(axis='y', labelcolor='purple')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    ratios = [l / (h + 1e-10) for l, h in zip(low_freq, high_freq)]
    ax.plot(steps, ratios, 'purple', linewidth=2, label='Low/High Ratio')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=30000, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_title('Frequency Dominance + Accuracy', fontsize=12, fontweight='bold')

    ax2 = ax.twinx()
    ax2.plot(steps, test_accs, 'g-', linewidth=2, alpha=0.5, label='Test Acc')
    ax2.set_ylabel('Test Accuracy', fontsize=11, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(fontsize=10, loc='lower right')

    # 子图 4: 归一化贡献（百分比）
    ax = axes[1, 1]
    total = np.array(low_freq) + np.array(high_freq)
    low_pct = np.array(low_freq) / (total + 1e-10) * 100
    high_pct = np.array(high_freq) / (total + 1e-10) * 100

    ax.plot(steps, low_pct, 'b-', linewidth=2, label='Low Freq %')
    ax.plot(steps, high_pct, 'r-', linewidth=2, label='High Freq %')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=30000, color='gray', linestyle='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('Normalized Contribution (%)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 添加 Grokking 区域标注
    for ax in axes.flat:
        ax.axvspan(30000, max(steps), alpha=0.1, color='yellow')

    plt.suptitle('Grokking: Frequency Band Evolution\\n'
                'Low frequency (k≤5) vs High frequency (k>5) contributions',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存图形
    output_file = os.path.join(output_dir, 'spectral_band_evolution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file}')

    output_file_pdf = os.path.join(output_dir, 'spectral_band_evolution.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file_pdf}')

    plt.close()


def plot_dominant_frequency_evolution(data, output_dir):
    """绘制主导频率演化"""
    steps = [int(row['step']) for row in data]

    # 提取前 10 个频率的数据
    max_freq = 10
    freq_data = {}
    for k in range(max_freq):
        col = f'freq_{k}_attribution'
        freq_data[k] = [float(row.get(col, 0)) for row in data]

    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # 子图 1: 所有频率曲线
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max_freq))

    for k in range(max_freq):
        ax.plot(steps, freq_data[k], linewidth=1.5, alpha=0.8,
               color=colors[k], label=f'k={k}')

    ax.set_xscale('log')
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Attribution (Energy)', fontsize=11)
    ax.set_title('Individual Frequency Evolution (k=0 to 9)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, ncol=5, loc='upper right')
    ax.grid(True, alpha=0.3)

    # 子图 2: 归一化到 [0, 1] 以便比较趋势
    ax = axes[1]
    for k in range(max_freq):
        values = np.array(freq_data[k])
        if values.max() > values.min():
            normalized = (values - values.min()) / (values.max() - values.min())
        else:
            normalized = np.zeros_like(values)
        ax.plot(steps, normalized, linewidth=1.5, alpha=0.8,
               color=colors[k], label=f'k={k}')

    ax.set_xscale('log')
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Normalized Attribution', fontsize=11)
    ax.set_title('Normalized Frequency Evolution (for trend comparison)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, ncol=5, loc='upper right')
    ax.grid(True, alpha=0.3)

    # 添加 Grokking 区域标注
    for ax in axes:
        ax.axvspan(30000, max(steps), alpha=0.1, color='yellow')

    plt.suptitle('Grokking: Dominant Frequency Analysis\\n'
                'Which frequencies drive the generalization?',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存图形
    output_file = os.path.join(output_dir, 'dominant_frequency_evolution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file}')

    output_file_pdf = os.path.join(output_dir, 'dominant_frequency_evolution.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file_pdf}')

    plt.close()


def main():
    spectral_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/spectral_attribution.csv"
    output_dir = "/root/data1/zjj/Grokking_Formulation/experiments/figures/2D_DFT"

    print("=" * 60)
    print("绘制频谱归因分析结果")
    print("=" * 60)
    print(f"频谱归因数据文件: {spectral_file}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    print("\n加载数据...")
    data = load_spectral_attribution_data(spectral_file)
    print(f"数据点数: {len(data)}")

    print("\n生成频谱归因热力图...")
    plot_spectral_heatmap(data, output_dir)

    print("\n生成频率带演化图...")
    plot_frequency_band_evolution(data, output_dir)

    print("\n生成主导频率演化图...")
    plot_dominant_frequency_evolution(data, output_dir)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
