#!/usr/bin/env python3
"""
绘制相位相干性随训练步数的变化

使用 phase_coherence.csv 和 metric.csv 绘制双 y 轴图：
- 左 y 轴：训练和测试准确率
- 右 y 轴：相位线性度 R²
- x 轴采用对数刻度
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_data(phase_coherence_file, metric_file):
    """加载相位相干性和 metric 数据"""
    # 加载相位相干性数据
    phase_coherence_data = []
    with open(phase_coherence_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase_coherence_data.append(row)

    # 加载 metric 数据
    metric_data = []
    with open(metric_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_data.append(row)

    return phase_coherence_data, metric_data


def plot_phase_coherence(phase_coherence_data, metric_data, output_dir):
    """绘制相位相干性图"""
    # 提取数据
    steps = [int(row['step']) for row in phase_coherence_data]
    train_accs = [float(row['train_acc']) for row in phase_coherence_data]
    test_accs = [float(row['test_acc']) for row in phase_coherence_data]
    mean_r2s = [float(row['phase_coherence_r2_mean']) for row in phase_coherence_data]
    median_r2s = [float(row['phase_coherence_r2_median']) for row in phase_coherence_data]
    r2_stds = [float(row['phase_coherence_r2_std']) for row in phase_coherence_data]
    r2_maxs = [float(row['phase_coherence_r2_max']) for row in phase_coherence_data]
    r2_mins = [float(row['phase_coherence_r2_min']) for row in phase_coherence_data]

    # 创建图形 - 双 y 轴主图
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # 绘制准确率曲线（左 y 轴）
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xscale('log')  # 对数刻度
    ax1.grid(True, alpha=0.3)

    line1, = ax1.plot(steps, train_accs, 'b-', label='Train Acc', linewidth=1.5, alpha=0.5)
    line2, = ax1.plot(steps, test_accs, 'r-', label='Test Acc', linewidth=2, alpha=0.7)
    ax1.legend(loc='upper left', fontsize=10)

    # 创建第二个 y 轴用于相位相干性
    ax2 = ax1.twinx()
    ax2.set_ylabel('Phase Linearity (R²)', fontsize=12, color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # 绘制均值 R² 和中位数 R²
    line3, = ax2.plot(steps, mean_r2s, 'orange', label='Mean R²',
                       linewidth=2, alpha=0.8)
    line4, = ax2.plot(steps, median_r2s, 'purple', linestyle='--',
                       label='Median R²', linewidth=2, alpha=0.8)

    # 添加第二个 y 轴的图例
    lines = [line3, line4]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right', fontsize=10)

    # 设置标题
    plt.title('Grokking: Accuracy & Phase Coherence (Phase Linearity)\n'
                 'R² measures linear correlation between phase φ(k) and frequency index k\n'
                 'Higher R² = more structured phase (φ ∝ k), Lower R² = random phase',
                 fontsize=14, fontweight='bold')

    # 添加 Grokking 区域标注
    grokking_start = 30000
    ax1.axvspan(grokking_start, max(steps), alpha=0.1, color='yellow', label='Grokking Phase')

    # 添加说明文字
    textstr = 'Phase Coherence Analysis:\n' \
               'DFT on each column of W_E\n' \
               'Linear regression: φ(k) ~ k\n' \
               'R² = phase linearity'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax1.text(0.02, 0.02, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)

    plt.tight_layout()

    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'Phase_Coherence.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file}')

    # 同时保存 PDF 版本
    output_file_pdf = os.path.join(output_dir, 'Phase_Coherence.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file_pdf}')

    plt.close()

    # 创建详细的多子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 子图 1: 双 y 轴主图
    ax = axes[0, 0]
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    line1, = ax.plot(steps, train_accs, 'b-', label='Train Acc', linewidth=1.5, alpha=0.5)
    line2, = ax.plot(steps, test_accs, 'r-', label='Test Acc', linewidth=2, alpha=0.7)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('Phase Coherence Overview', fontsize=12, fontweight='bold')

    ax2 = ax.twinx()
    ax2.set_ylabel('Phase Linearity (R²)', fontsize=11, color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    line3, = ax2.plot(steps, mean_r2s, 'orange', label='Mean R²', linewidth=2, alpha=0.8)
    line4, = ax2.plot(steps, median_r2s, 'purple', linestyle='--', label='Median R²', linewidth=2, alpha=0.8)
    lines = [line3, line4]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right', fontsize=9)

    # 子图 2: Mean R² 带标准差
    ax = axes[0, 1]
    ax.plot(steps, mean_r2s, 'orange', linewidth=2, label='Mean')
    ax.fill_between(steps, np.array(mean_r2s) - np.array(r2_stds),
                    np.array(mean_r2s) + np.array(r2_stds), alpha=0.3, color='orange', label='±1 Std')
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Mean R²', fontsize=11, color='tab:orange')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('Mean R² with Standard Deviation', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    # 子图 3: Min/Max R² 范围
    ax = axes[1, 0]
    ax.plot(steps, r2_maxs, 'g-', linewidth=1.5, label='Max R²', alpha=0.7)
    ax.plot(steps, r2_mins, 'r-', linewidth=1.5, label='Min R²', alpha=0.7)
    ax.fill_between(steps, r2_mins, r2_maxs, alpha=0.2, color='gray', label='Range')
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('R² Range', fontsize=11, color='tab:green')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('R² Range Across Columns', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    # 子图 4: R² 标准差
    ax = axes[1, 1]
    ax.plot(steps, r2_stds, 'brown', linewidth=2)
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('R² Standard Deviation', fontsize=11, color='tab:brown')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('R² Variability Across Embedding Dimensions', fontsize=12, fontweight='bold')

    # 添加 Grokking 区域标注
    for ax in axes.flat:
        grokking_start = 30000
        ax.axvspan(grokking_start, max(steps), alpha=0.1, color='yellow')

    plt.suptitle('Phase Coherence Analysis: Detailed View\n'
                 'Measuring phase linearity: φ(k) ∝ k',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存图形
    output_file2 = os.path.join(output_dir, 'Phase_Coherence_detailed.png')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file2}')

    output_file2_pdf = os.path.join(output_dir, 'Phase_Coherence_detailed.pdf')
    plt.savefig(output_file2_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file2_pdf}')

    plt.close()


def main():
    """主函数"""
    # 配置参数
    phase_coherence_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/phase_coherence.csv"
    metric_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/metric.csv"
    output_dir = "/root/data1/zjj/Grokking_Formulation/experiments/figures"

    print("=" * 60)
    print("绘制相位相干性变化图")
    print("=" * 60)
    print("相位相干性数据文件: {}".format(phase_coherence_file))
    print("Metric 数据文件: {}".format(metric_file))
    print("输出目录: {}".format(output_dir))
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    phase_coherence_data, metric_data = load_data(phase_coherence_file, metric_file)

    print("相位相干性数据点数: {}".format(len(phase_coherence_data)))
    print("Metric 数据点数: {}".format(len(metric_data)))

    # 绘制图形
    print("\n生成图形...")
    plot_phase_coherence(phase_coherence_data, metric_data, output_dir)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
