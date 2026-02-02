#!/usr/bin/env python3
"""
绘制内在维度随训练步数的变化

使用 intrinsic_dimension.csv 和 metric.csv 绘制双 y 轴图：
- 左 y 轴：训练和测试准确率
- 右 y 轴：内在维度、有效秩
- x 轴采用对数刻度
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_data(intrinsic_dim_file, metric_file):
    """加载内在维度和 metric 数据"""
    # 加载内在维度数据
    intrinsic_dim_data = []
    with open(intrinsic_dim_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            intrinsic_dim_data.append(row)

    # 加载 metric 数据
    metric_data = []
    with open(metric_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_data.append(row)

    return intrinsic_dim_data, metric_data


def plot_intrinsic_dimension(intrinsic_dim_data, metric_data, output_dir):
    """绘制内在维度图"""
    # 提取数据
    steps = [int(row['step']) for row in intrinsic_dim_data]
    train_accs = [float(row['train_acc']) for row in intrinsic_dim_data]
    test_accs = [float(row['test_acc']) for row in intrinsic_dim_data]
    id_W_Es = [float(row['id_W_E']) for row in intrinsic_dim_data]
    id_W_Us = [float(row['id_W_U']) for row in intrinsic_dim_data]
    eff_rank_W_QKs = [float(row['eff_rank_W_QK']) for row in intrinsic_dim_data]
    eff_rank_W_Es = [float(row['eff_rank_W_E']) for row in intrinsic_dim_data]
    eff_rank_W_Us = [float(row['eff_rank_W_U']) for row in intrinsic_dim_data]

    # 创建图形 - 多个子图显示不同指标
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ========== 子图 1: TwoNN 内在维度 ==========
    ax = axes[0, 0]
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    line1, = ax.plot(steps, train_accs, 'b-', label='Train Acc', linewidth=1.5, alpha=0.5)
    line2, = ax.plot(steps, test_accs, 'r-', label='Test Acc', linewidth=1.5, alpha=0.7)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('TwoNN Intrinsic Dimension', fontsize=12, fontweight='bold')

    ax2 = ax.twinx()
    ax2.set_ylabel('Intrinsic Dimension', fontsize=11, color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    line3, = ax2.plot(steps, id_W_Es, 'g-', label='ID (W_E)', linewidth=2, alpha=0.8)
    line4, = ax2.plot(steps, id_W_Us, 'm--', label='ID (W_U)', linewidth=2, alpha=0.8)
    lines = [line3, line4]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right', fontsize=9)

    # ========== 子图 2: W_QK 有效秩 ==========
    ax = axes[0, 1]
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    line1, = ax.plot(steps, train_accs, 'b-', label='Train Acc', linewidth=1.5, alpha=0.5)
    line2, = ax.plot(steps, test_accs, 'r-', label='Test Acc', linewidth=1.5, alpha=0.7)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('Effective Rank (W_QK = W_Q @ W_K^T)', fontsize=12, fontweight='bold')

    ax2 = ax.twinx()
    ax2.set_ylabel('Effective Rank', fontsize=11, color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    line3, = ax2.plot(steps, eff_rank_W_QKs, 'orange', label='EffRank (W_QK)', linewidth=2, alpha=0.8)
    ax2.legend([line3], [line3.get_label()], loc='upper right', fontsize=9)

    # ========== 子图 3: W_E 和 W_U 有效秩对比 ==========
    ax = axes[1, 0]
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    line1, = ax.plot(steps, train_accs, 'b-', label='Train Acc', linewidth=1.5, alpha=0.5)
    line2, = ax.plot(steps, test_accs, 'r-', label='Test Acc', linewidth=1.5, alpha=0.7)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('Effective Rank (W_E and W_U)', fontsize=12, fontweight='bold')

    ax2 = ax.twinx()
    ax2.set_ylabel('Effective Rank', fontsize=11, color='tab:purple')
    ax2.tick_params(axis='y', labelcolor='tab:purple')

    line3, = ax2.plot(steps, eff_rank_W_Es, 'purple', label='EffRank (W_E)', linewidth=2, alpha=0.8)
    line4, = ax2.plot(steps, eff_rank_W_Us, 'brown', linestyle='--', label='EffRank (W_U)', linewidth=2, alpha=0.8)
    lines = [line3, line4]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right', fontsize=9)

    # ========== 子图 4: 谱熵演化 ==========
    ax = axes[1, 1]
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    line1, = ax.plot(steps, train_accs, 'b-', label='Train Acc', linewidth=1.5, alpha=0.5)
    line2, = ax.plot(steps, test_accs, 'r-', label='Test Acc', linewidth=1.5, alpha=0.7)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('Spectral Entropy', fontsize=12, fontweight='bold')

    ax2 = ax.twinx()
    ax2.set_ylabel('Spectral Entropy', fontsize=11, color='tab:cyan')
    ax2.tick_params(axis='y', labelcolor='tab:cyan')

    entropy_W_Es = [float(row['entropy_W_E']) for row in intrinsic_dim_data]
    entropy_W_Us = [float(row['entropy_W_U']) for row in intrinsic_dim_data]
    entropy_W_QKs = [float(row['entropy_W_QK']) for row in intrinsic_dim_data]

    line3, = ax2.plot(steps, entropy_W_Es, 'c-', label='Entropy (W_E)', linewidth=1.5, alpha=0.8)
    line4, = ax2.plot(steps, entropy_W_Us, 'm--', label='Entropy (W_U)', linewidth=1.5, alpha=0.8)
    line5, = ax2.plot(steps, entropy_W_QKs, 'orange', linestyle=':', label='Entropy (W_QK)', linewidth=2, alpha=0.8)
    lines = [line3, line4, line5]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right', fontsize=8)

    # 添加 Grokking 区域标注
    for ax in axes.flat:
        if ax in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]:
            grokking_start = 30000
            ax.axvspan(grokking_start, max(steps), alpha=0.1, color='yellow')

    plt.suptitle('Grokking: Intrinsic Dimension & Effective Rank Evolution\n'
                 'TwoNN ID estimates data manifold dimension | Effective rank measures matrix complexity',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'intrinsic_dimension.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file}')

    # 同时保存 PDF 版本
    output_file_pdf = os.path.join(output_dir, 'intrinsic_dimension.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file_pdf}')

    plt.close()

    # 创建单独的综合图
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # 绘制准确率（左 y 轴）
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    line1, = ax1.plot(steps, train_accs, 'b-', label='Train Acc', linewidth=1.5, alpha=0.5)
    line2, = ax1.plot(steps, test_accs, 'r-', label='Test Acc', linewidth=2, alpha=0.7)
    ax1.legend(loc='upper left')

    # 创建第二个 y 轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('Intrinsic Dimension / Effective Rank', fontsize=12, color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    line3, = ax2.plot(steps, id_W_Es, 'g-', label='ID (W_E)', linewidth=2, alpha=0.8)
    line4, = ax2.plot(steps, id_W_Us, 'm--', label='ID (W_U)', linewidth=2, alpha=0.8)
    line5, = ax2.plot(steps, eff_rank_W_QKs, 'orange', linestyle=':', label='EffRank (W_QK)', linewidth=2.5, alpha=0.9)

    lines = [line3, line4, line5]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right')

    # 添加 Grokking 区域标注
    grokking_start = 30000
    ax1.axvspan(grokking_start, max(steps), alpha=0.1, color='yellow', label='Grokking Phase')

    plt.title('Grokking: Accuracy & Intrinsic Dimension Evolution\n'
              'TwoNN Intrinsic Dimension estimates the data manifold dimension\n'
              'Effective Rank of W_QK measures the attention complexity',
              fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 保存图形
    output_file2 = os.path.join(output_dir, 'intrinsic_dimension_combined.png')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file2}')

    output_file2_pdf = os.path.join(output_dir, 'intrinsic_dimension_combined.pdf')
    plt.savefig(output_file2_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file2_pdf}')

    plt.close()


def main():
    """主函数"""
    # 配置参数
    intrinsic_dim_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/intrinsic_dimension.csv"
    metric_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/metric.csv"
    output_dir = "/root/data1/zjj/Grokking_Formulation/experiments/figures"

    print("=" * 60)
    print("绘制内在维度变化图")
    print("=" * 60)
    print("内在维度数据文件: {}".format(intrinsic_dim_file))
    print("Metric 数据文件: {}".format(metric_file))
    print("输出目录: {}".format(output_dir))
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    intrinsic_dim_data, metric_data = load_data(intrinsic_dim_file, metric_file)

    print("内在维度数据点数: {}".format(len(intrinsic_dim_data)))
    print("Metric 数据点数: {}".format(len(metric_data)))

    # 绘制图形
    print("\n生成图形...")
    plot_intrinsic_dimension(intrinsic_dim_data, metric_data, output_dir)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
