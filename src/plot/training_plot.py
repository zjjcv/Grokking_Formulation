#!/usr/bin/env python3
"""
绘制 Grokking 实验的训练曲线图
使用 metric.csv 数据生成美观的可视化图表
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import os

# 设置中文字体和样式
rcParams['font.family'] = 'DejaVu Sans'
rcParams['figure.facecolor'] = 'white'
rcParams['axes.facecolor'] = '#f8f9fa'
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.3
rcParams['axes.edgecolor'] = '#dee2e6'

# 颜色方案
COLORS = {
    'train_loss': '#e63946',
    'train_acc': '#457b9d',
    'test_loss': '#f4a261',
    'test_acc': '#2a9d8f',
}


def smooth_curve(data, window=500):
    """平滑曲线"""
    if len(data) < window:
        return data
    return pd.Series(data).rolling(window=window, min_periods=1, center=True).mean().values


def plot_training_curves(df, output_dir):
    """绘制训练曲线图 - 包含训练/测试的 acc 和 loss"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 提取数据
    steps = df['step'].values
    train_loss = df['train_loss'].values
    train_acc = df['train_acc'].values
    test_loss = df['test_loss'].values
    test_acc = df['test_acc'].values

    # ========== 左图: 准确率曲线 ==========
    ax = axes[0]

    # 绘制原始曲线
    ax.plot(steps, train_acc, color=COLORS['train_acc'], linewidth=1.5, linestyle='--', alpha=0.7, label='Train Accuracy')
    ax.plot(steps, test_acc, color=COLORS['test_acc'], linewidth=1.5, label='Test Accuracy')

    # 标记峰值
    max_test_idx = np.argmax(test_acc)
    max_test_step = steps[max_test_idx]
    max_test_acc = test_acc[max_test_idx]
    ax.scatter([max_test_step], [max_test_acc], color=COLORS['test_acc'], s=150, zorder=5, edgecolors='white', linewidth=1.5)
    ax.annotate(f'Peak: {max_test_acc:.1%}\n@ step {max_test_step:,}',
                xy=(max_test_step, max_test_acc), xytext=(max_test_step * 0.5, max_test_acc + 0.15),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=COLORS['test_acc'], alpha=0.9),
                arrowprops=dict(arrowstyle='->', color=COLORS['test_acc'], lw=1.5))

    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim(-0.02, 1.05)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')

    # ========== 右图: 损失曲线 ==========
    ax = axes[1]

    # 绘制原始曲线
    ax.plot(steps, train_loss, color=COLORS['train_loss'], linewidth=1.5, linestyle='--', alpha=0.7, label='Train Loss')
    ax.plot(steps, test_loss, color=COLORS['test_loss'], linewidth=1.5, label='Test Loss')

    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')

    # ========== 总标题和信息 ==========
    final_train_acc = train_acc[-1]
    final_test_acc = test_acc[-1]
    max_test_acc = np.max(test_acc)

    info_text = f'Final Train Acc: {final_train_acc:.1%} | Final Test Acc: {final_test_acc:.1%} | Peak Test Acc: {max_test_acc:.1%}'
    fig.suptitle('Grokking: x + y (mod 97), wd=0.005', fontsize=15, fontweight='bold')
    fig.text(0.5, 0.02, info_text, fontsize=11, ha='center', style='italic',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#e3f2fd', edgecolor='#90caf9', alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 0.93])

    # 保存
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'training_curves.pdf'), bbox_inches='tight', facecolor='white')
    print(f"图表已保存至: {output_dir}/training_curves.png")
    plt.close()


def main():
    """主函数"""
    metric_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/metric.csv"
    output_dir = "/root/data1/zjj/Grokking_Formulation/experiments/figures"

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("绘制 Grokking 训练曲线")
    print("=" * 60)

    # 加载数据
    print(f"加载数据: {metric_file}")
    df = pd.read_csv(metric_file)
    print(f"数据点数: {len(df)}")

    # 打印统计信息
    print("\n=== 训练统计 ===")
    print(f"总训练步数: {df['step'].max():,}")
    print(f"最终训练准确率: {df['train_acc'].iloc[-1]:.2%}")
    print(f"最终测试准确率: {df['test_acc'].iloc[-1]:.2%}")
    print(f"最高测试准确率: {df['test_acc'].max():.2%} (Step {df.loc[df['test_acc'].idxmax(), 'step']})")

    # 生成图表
    print("\n生成图表...")
    plot_training_curves(df, output_dir)

    print("\n" + "=" * 60)
    print("图表已生成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
