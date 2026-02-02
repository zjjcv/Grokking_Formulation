#!/usr/bin/env python3
"""
绘制 2D 频域基尼系数和逆参与率随训练步数的变化

使用 gini_ip_2d.csv 和 metric.csv 绘制三 y 轴图：
- 左 y 轴：训练和测试准确率
- 右 y 轴：基尼系数和逆参与率（2D DFT 版本）
- x 轴采用对数刻度
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_data(gini_ip_file, metric_file):
    """加载 Gini/IPR 和 metric 数据"""
    gini_ip_data = []
    with open(gini_ip_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gini_ip_data.append(row)

    metric_data = []
    with open(metric_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_data.append(row)

    return gini_ip_data, metric_data


def plot_gini_ipr_2d(gini_ip_data, metric_data, output_dir):
    """绘制基尼系数和逆参与率图（2D DFT 版本）"""
    steps = [int(row['step']) for row in gini_ip_data]
    train_accs = [float(row['train_acc']) for row in gini_ip_data]
    test_accs = [float(row['test_acc']) for row in gini_ip_data]
    ginis = [float(row['gini']) for row in gini_ip_data]
    iprs = [float(row['ipr']) for row in gini_ip_data]

    # 创建图形
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # 绘制准确率曲线（左 y 轴）
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xscale('log')  # 对数刻度
    ax1.grid(True, alpha=0.3)

    line1, = ax1.plot(steps, train_accs, 'b-', label='Train Acc', linewidth=1.5, alpha=0.7)
    line2, = ax1.plot(steps, test_accs, 'r-', label='Test Acc', linewidth=1.5, alpha=0.7)
    ax1.legend(loc='upper left')

    # 创建第二个 y 轴用于基尼系数
    ax2 = ax1.twinx()
    ax2.set_ylabel('2D Frequency Domain Metrics', fontsize=12, color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # 绘制基尼系数
    line3, = ax2.plot(steps, ginis, 'g-', label='Gini Coefficient (2D)', linewidth=2, alpha=0.8)
    # 绘制逆参与率（归一化）
    iprs_normalized = [(ipr - min(iprs)) / (max(iprs) - min(iprs)) for ipr in iprs]
    line4, = ax2.plot(steps, iprs_normalized, 'm--', label='IPR (normalized)', linewidth=2, alpha=0.8)

    lines = [line3, line4]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')

    # 设置标题
    plt.title('Grokking: Accuracy, Gini Coefficient & IPR (2D DFT)\n'
                 '2D DFT: FFT2 on embedding matrix captures bidirectional frequency patterns\n'
                 'Gini: inequality of 2D frequency distribution | IPR: effective dimensionality',
                 fontsize=14, fontweight='bold')

    # 添加 Grokking 区域标注
    grokking_start = 30000
    ax1.axvspan(grokking_start, max(steps), alpha=0.1, color='yellow', label='Grokking Phase')

    # 添加说明文字
    textstr = '2D DFT Analysis:\n' \
               'Captures both token and feature dimensions\n' \
               'Gini Coeff: Lower = more unequal freq distribution\n' \
               'IPR: Higher = more diverse frequency components'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax1.text(0.02, 0.02, textstr, transform=ax1.transAxes, fontsize=8,
            verticalalignment='bottom', bbox=props)

    plt.tight_layout()

    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'gini_ipr_2d.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file}')

    output_file_pdf = os.path.join(output_dir, 'gini_ipr_2d.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file_pdf}')

    plt.close()

    # 创建详细的多子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 子图 1: Gini 系数单独显示
    ax = axes[0, 0]
    ax.plot(steps, ginis, 'g-', linewidth=2)
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Gini Coefficient', fontsize=11, color='tab:green')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('Gini Coefficient (2D DFT)', fontsize=12, fontweight='bold')

    # 子图 2: IPR 单独显示
    ax = axes[0, 1]
    ax.plot(steps, iprs, 'purple', linewidth=2)
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Inverse Participation Ratio', fontsize=11, color='tab:purple')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('IPR (2D DFT)', fontsize=12, fontweight='bold')

    # 子图 3: Gini 变化率
    ax = axes[1, 0]
    ginis_diff = np.diff(ginis)
    ax.plot(steps[1:], ginis_diff, 'brown', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Gini Change Rate', fontsize=11)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('Gini Coefficient Change Rate', fontsize=12, fontweight='bold')

    # 子图 4: IPR 变化率
    ax = axes[1, 1]
    iprs_diff = np.diff(iprs)
    ax.plot(steps[1:], iprs_diff, 'orange', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('IPR Change Rate', fontsize=11)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('IPR Change Rate', fontsize=12, fontweight='bold')

    # 添加 Grokking 区域标注
    for ax in axes.flat:
        grokking_start = 30000
        ax.axvspan(grokking_start, max(steps), alpha=0.1, color='yellow')

    plt.suptitle('2D DFT Analysis: Detailed View',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存图形
    output_file2 = os.path.join(output_dir, 'gini_ipr_2d_detailed.png')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file2}')

    output_file2_pdf = os.path.join(output_dir, 'gini_ipr_2d_detailed.pdf')
    plt.savefig(output_file2_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file2_pdf}')

    plt.close()


def main():
    gini_ip_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/gini_ip_2d.csv"
    metric_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/metric.csv"
    output_dir = "/root/data1/zjj/Grokking_Formulation/experiments/figures/2D_DFT"

    print("=" * 60)
    print("绘制 2D 频域基尼系数和逆参与率变化图")
    print("=" * 60)
    print(f"Gini/IPR 数据文件: {gini_ip_file}")
    print(f"Metric 数据文件: {metric_file}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    print("\n加载数据...")
    gini_ip_data, metric_data = load_data(gini_ip_file, metric_file)
    print(f"Gini/IPR 数据点数: {len(gini_ip_data)}")
    print(f"Metric 数据点数: {len(metric_data)}")

    print("\n生成图形...")
    plot_gini_ipr_2d(gini_ip_data, metric_data, output_dir)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
