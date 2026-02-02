#!/usr/bin/env python3
"""
绘制 2D 频域对齐随训练步数的变化

使用 f_alginment_2d.csv 和 metric.csv 绘制双 y 轴图：
- 左 y 轴：训练和测试准确率
- 右 y 轴：2D 频域对齐度（余弦相似度）
- x 轴采用对数刻度
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_data(f_alginment_file, metric_file):
    """加载频域对齐和 metric 数据"""
    f_alginment_data = []
    with open(f_alginment_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            f_alginment_data.append(row)

    metric_data = []
    with open(metric_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_data.append(row)

    return f_alginment_data, metric_data


def plot_2d_f_alginment(f_alginment_data, metric_data, output_dir):
    """绘制 2D 频域对齐图"""
    steps = [int(row['step']) for row in f_alginment_data]
    train_accs = [float(row['train_acc']) for row in f_alginment_data]
    test_accs = [float(row['test_acc']) for row in f_alginment_data]
    alignments = [float(row['mean_2d_alignment']) for row in f_alginment_data]
    w_e_energies = [float(row['w_e_energy']) for row in f_alginment_data]
    w_u_energies = [float(row['w_u_energy']) for row in f_alginment_data]

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 子图 1: 主图 - 双 y 轴
    ax = axes[0, 0]
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    line1, = ax.plot(steps, train_accs, 'b-', label='Train Acc', linewidth=1.5, alpha=0.5)
    line2, = ax.plot(steps, test_accs, 'r-', label='Test Acc', linewidth=2, alpha=0.7)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title('2D Frequency Alignment Overview', fontsize=12, fontweight='bold')

    ax2 = ax.twinx()
    ax2.set_ylabel('2D Alignment (Cosine Sim)', fontsize=12, color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    line3, = ax2.plot(steps, alignments, 'orange', linewidth=2, alpha=0.8)
    ax2.legend([line3], [line3.get_label()], loc='center right', fontsize=10)

    # 子图 2: 能量对比
    ax = axes[0, 1]
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Energy', fontsize=11, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    line1, = ax.plot(steps, w_e_energies, 'b-', label='W_E Energy', linewidth=2, alpha=0.7)
    line2, = ax.plot(steps, w_u_energies, 'r-', label='W_U Energy', linewidth=2, alpha=0.7)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title('2D Frequency Energy', fontsize=12, fontweight='bold')

    # 子图 3: 对齐度变化率
    ax = axes[1, 0]
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Alignment', fontsize=11)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    ax.plot(steps, alignments, 'orange', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline (0.5)')
    ax.legend(fontsize=10)
    ax.set_title('2D Alignment (Evolution)', fontsize=12, fontweight='bold')

    # 子图 4: 对齐度 vs 准确率散点图
    ax = axes[1, 1]
    ax.scatter(test_accs, alignments, c=steps, cmap='viridis', alpha=0.6)
    ax.set_xlabel('Test Accuracy', fontsize=11)
    ax.set_ylabel('2D Alignment', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title('Alignment vs Accuracy', fontsize=12, fontweight='bold')

    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Training Step', fontsize=10)

    # 添加 Grokking 区域标注
    for ax in axes.flat:
        grokking_start = 30000
        ax.axvspan(grokking_start, max(steps), alpha=0.1, color='yellow')

    plt.suptitle('Grokking: 2D Frequency Domain Alignment\n'
                 'FFT2 on W_E and W_U, then cosine similarity',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'f_alginment_2d.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file}')

    output_file_pdf = os.path.join(output_dir, 'f_alginment_2d.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file_pdf}')

    plt.close()


def main():
    f_alginment_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/f_alginment_2d.csv"
    metric_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/metric.csv"
    output_dir = "/root/data1/zjj/Grokking_Formulation/experiments/figures/2D_DFT"

    print("=" * 60)
    print("绘制 2D 频域对齐变化图")
    print("=" * 60)
    print(f"频域对齐数据文件: {f_alginment_file}")
    print(f"Metric 数据文件: {metric_file}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    print("\n加载数据...")
    f_alginment_data, metric_data = load_data(f_alginment_file, metric_file)
    print(f"频域对齐数据点数: {len(f_alginment_data)}")
    print(f"Metric 数据点数: {len(metric_data)}")

    print("\n生成图形...")
    plot_2d_f_alginment(f_alginment_data, metric_data, output_dir)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
