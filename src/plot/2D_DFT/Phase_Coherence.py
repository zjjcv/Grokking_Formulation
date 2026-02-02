#!/usr/bin/env python3
"""
绘制 2D 框架下的相位相干性随训练步数的变化

使用 phase_coherence_2d.csv 和 metric.csv 绘制双 y 轴图：
- 左 y 轴：训练和测试准确率
- 右 y 轴：2D 相位线性度 R²（行方向分析）
- x 轴采用对数刻度
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_data(phase_coherence_file, metric_file):
    """加载相位相干性和 metric 数据"""
    phase_coherence_data = []
    with open(phase_coherence_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase_coherence_data.append(row)

    metric_data = []
    with open(metric_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_data.append(row)

    return phase_coherence_data, metric_data


def plot_2d_phase_coherence(phase_coherence_data, metric_data, output_dir):
    """绘制 2D 相位相干性图"""
    steps = [int(row['step']) for row in phase_coherence_data]
    train_accs = [float(row['train_acc']) for row in phase_coherence_data]
    test_accs = [float(row['test_acc']) for row in phase_coherence_data]
    mean_r2s = [float(row['phase_coherence_r2_mean']) for row in phase_coherence_data]
    r2_stds = [float(row['phase_coherence_r2_std']) for row in phase_coherence_data]

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 子图 1: 双 y 轴主图
    ax = axes[0, 0]
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    line1, = ax.plot(steps, train_accs, 'b-', label='Train Acc', linewidth=1.5, alpha=0.5)
    line2, = ax.plot(steps, test_accs, 'r-', label='Test Acc', linewidth=2, alpha=0.7)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title('2D Phase Coherence Overview', fontsize=12, fontweight='bold')

    ax2 = ax.twinx()
    ax2.set_ylabel('Phase Linearity (R²)', fontsize=12, color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    line3, = ax2.plot(steps, mean_r2s, 'orange', linewidth=2, alpha=0.8)
    ax2.legend([line3], [line3.get_label()], loc='center right', fontsize=10)

    # 子图 2: Mean R² 带标准差
    ax = axes[0, 1]
    ax.plot(steps, mean_r2s, 'orange', linewidth=2, label='Mean R²')
    ax.fill_between(steps, np.array(mean_r2s) - np.array(r2_stds),
                    np.array(mean_r2s) + np.array(r2_stds), alpha=0.3, color='orange')
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('R²', fontsize=11, color='tab:orange')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('2D Phase Linearity (Row-wise DFT)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    # 子图 3: R² vs 准确率散点
    ax = axes[1, 0]
    sc = ax.scatter(test_accs, mean_r2s, c=steps, cmap='viridis', alpha=0.6)
    ax.set_xlabel('Test Accuracy', fontsize=11)
    ax.set_ylabel('Mean R²', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title('Phase Linearity vs Accuracy', fontsize=12, fontweight='bold')

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Training Step', fontsize=10)

    # 子图 4: R² 标准差
    ax = axes[1, 1]
    ax.plot(steps, r2_stds, 'brown', linewidth=2)
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('R² Standard Deviation', fontsize=11)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('2D Phase Linearity Variability', fontsize=12, fontweight='bold')

    # 添加 Grokking 区域标注
    for ax in axes.flat:
        grokking_start = 30000
        ax.axvspan(grokking_start, max(steps), alpha=0.1, color='yellow')

    plt.suptitle('Grokking: 2D Phase Coherence Analysis\n'
                 'Row-wise DFT on embedding matrix | R² measures phase linearity',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'Phase_Coherence_2d.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file}')

    output_file_pdf = os.path.join(output_dir, 'Phase_Coherence_2d.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file_pdf}')

    plt.close()


def main():
    phase_coherence_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/phase_coherence_2d.csv"
    metric_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/metric.csv"
    output_dir = "/root/data1/zjj/Grokking_Formulation/experiments/figures/2D_DFT"

    print("=" * 60)
    print("绘制 2D 框架下的相位相干性变化图")
    print("=" * 60)
    print(f"相位相干性数据文件: {phase_coherence_file}")
    print(f"Metric 数据文件: {metric_file}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    print("\n加载数据...")
    phase_coherence_data, metric_data = load_data(phase_coherence_file, metric_file)
    print(f"相位相干性数据点数: {len(phase_coherence_data)}")
    print(f"Metric 数据点数: {len(metric_data)}")

    print("\n生成图形...")
    plot_2d_phase_coherence(phase_coherence_data, metric_data, output_dir)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
