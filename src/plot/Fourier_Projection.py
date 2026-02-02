#!/usr/bin/env python3
"""
绘制空间基与傅里叶基的投影稀疏度随训练步数的变化

使用 fourier_projection.csv 和 metric.csv 绘图：
- 展示空间域稀疏度与频域稀疏度的消长关系
- 验证 IPR 下降是否对应频域稀疏度上升
- x 轴采用对数刻度
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_data(fourier_projection_file, metric_file):
    """加载傅里叶投影和 metric 数据"""
    # 加载傅里叶投影数据
    fourier_projection_data = []
    with open(fourier_projection_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fourier_projection_data.append(row)

    # 加载 metric 数据
    metric_data = []
    with open(metric_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_data.append(row)

    return fourier_projection_data, metric_data


def plot_fourier_projection(fourier_projection_data, metric_data, output_dir):
    """绘制傅里叶投影稀疏度图"""
    # 提取数据
    steps = [int(row['step']) for row in fourier_projection_data]
    train_accs = [float(row['train_acc']) for row in fourier_projection_data]
    test_accs = [float(row['test_acc']) for row in fourier_projection_data]

    # W_E 数据
    spatial_l1l2_E = [float(row['W_E_spatial_l1l2']) for row in fourier_projection_data]
    fourier_l1l2_E = [float(row['W_E_fourier_l1l2']) for row in fourier_projection_data]
    spatial_gini_E = [float(row['W_E_spatial_gini']) for row in fourier_projection_data]
    fourier_gini_E = [float(row['W_E_fourier_gini']) for row in fourier_projection_data]

    # W_U 数据
    spatial_l1l2_U = [float(row['W_U_spatial_l1l2']) for row in fourier_projection_data]
    fourier_l1l2_U = [float(row['W_U_fourier_l1l2']) for row in fourier_projection_data]
    spatial_gini_U = [float(row['W_U_spatial_gini']) for row in fourier_projection_data]
    fourier_gini_U = [float(row['W_U_fourier_gini']) for row in fourier_projection_data]

    # ========== 图1: 消长关系主图（L1/L2） ==========
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # W_E 消长关系
    ax = axes[0]
    ax.set_ylabel('Sparsity (L1/L2)', fontsize=12, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    line1, = ax.plot(steps, spatial_l1l2_E, 'b-', linewidth=2, label='Spatial Domain', alpha=0.8)
    line2, = ax.plot(steps, fourier_l1l2_E, 'r-', linewidth=2, label='Fourier Domain', alpha=0.8)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title('W_E: Spatial vs Fourier Sparsity (L1/L2)', fontsize=12, fontweight='bold')

    # 右 y 轴：准确率
    ax_r = ax.twinx()
    ax_r.plot(steps, test_accs, 'g:', linewidth=1.5, alpha=0.5, label='Test Acc')
    ax_r.set_ylabel('Accuracy', fontsize=11)
    ax_r.legend(loc='upper right', fontsize=10)

    # W_U 消长关系
    ax = axes[1]
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Sparsity (L1/L2)', fontsize=12, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    line3, = ax.plot(steps, spatial_l1l2_U, 'b-', linewidth=2, label='Spatial Domain', alpha=0.8)
    line4, = ax.plot(steps, fourier_l1l2_U, 'r-', linewidth=2, label='Fourier Domain', alpha=0.8)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title('W_U: Spatial vs Fourier Sparsity (L1/L2)', fontsize=12, fontweight='bold')

    # 右 y 轴：准确率
    ax_r = ax.twinx()
    ax_r.plot(steps, test_accs, 'g:', linewidth=1.5, alpha=0.5, label='Test Acc')
    ax_r.set_ylabel('Accuracy', fontsize=11)
    ax_r.legend(loc='upper right', fontsize=10)

    # 添加 Grokking 区域标注
    for ax in axes:
        grokking_start = 30000
        ax.axvspan(grokking_start, max(steps), alpha=0.1, color='yellow')

    plt.suptitle('Grokking: Spatial vs Fourier Domain Sparsity Trade-off\n'
                 'Lower L1/L2 = sparser representation',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'Fourier_Projection_l1l2.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file}')

    output_file_pdf = os.path.join(output_dir, 'Fourier_Projection_l1l2.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file_pdf}')

    plt.close()

    # ========== 图2: Gini 系数消长关系 ==========
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # W_E Gini 消长
    ax = axes[0]
    ax.set_ylabel('Gini Coefficient', fontsize=12, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    ax.plot(steps, spatial_gini_E, 'b-', linewidth=2, label='Spatial Domain', alpha=0.8)
    ax.plot(steps, fourier_gini_E, 'r-', linewidth=2, label='Fourier Domain', alpha=0.8)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title('W_E: Spatial vs Fourier Gini Coefficient', fontsize=12, fontweight='bold')

    ax_r = ax.twinx()
    ax_r.plot(steps, test_accs, 'g:', linewidth=1.5, alpha=0.5, label='Test Acc')
    ax_r.set_ylabel('Accuracy', fontsize=11)
    ax_r.legend(loc='upper right', fontsize=10)

    # W_U Gini 消长
    ax = axes[1]
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Gini Coefficient', fontsize=12, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    ax.plot(steps, spatial_gini_U, 'b-', linewidth=2, label='Spatial Domain', alpha=0.8)
    ax.plot(steps, fourier_gini_U, 'r-', linewidth=2, label='Fourier Domain', alpha=0.8)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title('W_U: Spatial vs Fourier Gini Coefficient', fontsize=12, fontweight='bold')

    ax_r = ax.twinx()
    ax_r.plot(steps, test_accs, 'g:', linewidth=1.5, alpha=0.5, label='Test Acc')
    ax_r.set_ylabel('Accuracy', fontsize=11)
    ax_r.legend(loc='upper right', fontsize=10)

    # 添加 Grokking 区域标注
    for ax in axes:
        grokking_start = 30000
        ax.axvspan(grokking_start, max(steps), alpha=0.1, color='yellow')

    plt.suptitle('Grokking: Spatial vs Fourier Domain Gini Coefficient\n'
                 'Higher Gini = more unequal distribution (sparser)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    # 保存图形
    output_file2 = os.path.join(output_dir, 'Fourier_Projection_gini.png')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file2}')

    output_file2_pdf = os.path.join(output_dir, 'Fourier_Projection_gini.pdf')
    plt.savefig(output_file2_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file2_pdf}')

    plt.close()

    # ========== 图3: 详细多子图 ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 子图 1: W_U L1/L2 消长
    ax = axes[0, 0]
    ax.plot(steps, spatial_l1l2_U, 'b-', linewidth=2, label='Spatial')
    ax.plot(steps, fourier_l1l2_U, 'r-', linewidth=2, label='Fourier')
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('L1/L2 Sparsity', fontsize=11)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('W_U: L1/L2 Trade-off', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    # 子图 2: W_U Gini 消长
    ax = axes[0, 1]
    ax.plot(steps, spatial_gini_U, 'b-', linewidth=2, label='Spatial')
    ax.plot(steps, fourier_gini_U, 'r-', linewidth=2, label='Fourier')
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Gini Coefficient', fontsize=11)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('W_U: Gini Trade-off', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    # 子图 3: 稀疏度差值
    ax = axes[1, 0]
    diff_l1l2 = np.array(fourier_l1l2_U) - np.array(spatial_l1l2_U)
    ax.plot(steps, diff_l1l2, 'purple', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(steps, 0, diff_l1l2, where=np.array(diff_l1l2)>0,
                    alpha=0.3, color='red', label='Fourier > Spatial')
    ax.fill_between(steps, 0, diff_l1l2, where=np.array(diff_l1l2)<0,
                    alpha=0.3, color='blue', label='Spatial > Fourier')
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Fourier - Spatial (L1/L2)', fontsize=11)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('Sparsity Difference (L1/L2)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    # 子图 4: Gini 差值
    ax = axes[1, 1]
    diff_gini = np.array(fourier_gini_U) - np.array(spatial_gini_U)
    ax.plot(steps, diff_gini, 'purple', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(steps, 0, diff_gini, where=np.array(diff_gini)>0,
                    alpha=0.3, color='red', label='Fourier > Spatial')
    ax.fill_between(steps, 0, diff_gini, where=np.array(diff_gini)<0,
                    alpha=0.3, color='blue', label='Spatial > Fourier')
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Fourier - Spatial (Gini)', fontsize=11)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('Gini Difference', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    # 添加 Grokking 区域标注
    for ax in axes.flat:
        grokking_start = 30000
        ax.axvspan(grokking_start, max(steps), alpha=0.1, color='yellow')

    plt.suptitle('Fourier Projection: Detailed Trade-off Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存图形
    output_file3 = os.path.join(output_dir, 'Fourier_Projection_detailed.png')
    plt.savefig(output_file3, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file3}')

    output_file3_pdf = os.path.join(output_dir, 'Fourier_Projection_detailed.pdf')
    plt.savefig(output_file3_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file3_pdf}')

    plt.close()


def main():
    """主函数"""
    # 配置参数
    fourier_projection_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/fourier_projection.csv"
    metric_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/metric.csv"
    output_dir = "/root/data1/zjj/Grokking_Formulation/experiments/figures"

    print("=" * 60)
    print("绘制空间基与傅里叶基的投影稀疏度变化图")
    print("=" * 60)
    print("傅里叶投影数据文件: {}".format(fourier_projection_file))
    print("Metric 数据文件: {}".format(metric_file))
    print("输出目录: {}".format(output_dir))
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    fourier_projection_data, metric_data = load_data(fourier_projection_file, metric_file)

    print("傅里叶投影数据点数: {}".format(len(fourier_projection_data)))
    print("Metric 数据点数: {}".format(len(metric_data)))

    # 绘制图形
    print("\n生成图形...")
    plot_fourier_projection(fourier_projection_data, metric_data, output_dir)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
