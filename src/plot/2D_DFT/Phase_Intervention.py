#!/usr/bin/env python3
"""
绘制相位干预实验结果

使用 phase_intervention.csv 绘制：
- 原始测试准确率 vs 训练步数
- 干预后测试准确率 vs 训练步数
- 准确率提升曲线

预期：在 Grokking 前的平台期，干预后准确率应显著高于原始准确率
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_data(phase_intervention_file):
    """加载相位干预数据"""
    data = []
    with open(phase_intervention_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def plot_phase_intervention(data, output_dir):
    """绘制相位干预实验结果"""
    steps = [int(row['step']) for row in data]
    train_accs = [float(row['train_acc']) for row in data]
    original_test_accs = [float(row['original_test_acc']) for row in data]
    intervened_test_accs = [float(row['intervened_test_acc']) for row in data]
    improvements = [float(row['acc_improvement']) for row in data]

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 子图 1: 原始 vs 干预准确率对比（主图）
    ax = axes[0, 0]
    ax.plot(steps, original_test_accs, 'b-', linewidth=2, label='Original Test Acc', alpha=0.7)
    ax.plot(steps, intervened_test_accs, 'r-', linewidth=2, label='Intervened Test Acc', alpha=0.7)
    ax.axvline(x=30000, color='gray', linestyle='--', alpha=0.5, label='Grokking Start')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_title('Phase Intervention: Original vs Intervened Accuracy', fontsize=12, fontweight='bold')

    # 子图 2: 准确率提升
    ax = axes[0, 1]
    ax.plot(steps, improvements, 'purple', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.fill_between(steps, 0, improvements, where=np.array(improvements) > 0,
                     alpha=0.3, color='green', label='Positive Improvement')
    ax.fill_between(steps, 0, improvements, where=np.array(improvements) < 0,
                     alpha=0.3, color='red', label='Negative Impact')
    ax.axvline(x=30000, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Accuracy Improvement', fontsize=12)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_title('Phase Intervention Effect (Intervened - Original)', fontsize=12, fontweight='bold')

    # 子图 3: 早期阶段放大图（线性刻度）
    ax = axes[1, 0]
    # 只显示前 30000 步
    early_mask = np.array(steps) <= 30000
    early_steps = np.array(steps)[early_mask]
    early_original = np.array(original_test_accs)[early_mask]
    early_intervened = np.array(intervened_test_accs)[early_mask]

    ax.plot(early_steps, early_original, 'b-', linewidth=2, label='Original', alpha=0.7)
    ax.plot(early_steps, early_intervened, 'r-', linewidth=2, label='Intervened', alpha=0.7)
    ax.fill_between(early_steps, early_original, early_intervened,
                     where=np.array(early_intervened) > np.array(early_original),
                     alpha=0.3, color='green', label='Intervention Wins')
    ax.fill_between(early_steps, early_original, early_intervened,
                     where=np.array(early_intervened) <= np.array(early_original),
                     alpha=0.3, color='gray')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_title('Pre-Grokking Phase (Linear Scale)', fontsize=12, fontweight='bold')

    # 子图 4: 干预效果热力图（阶段分析）
    ax = axes[1, 1]

    # 定义训练阶段
    phases = {
        'Memorization\n(0-5K)': (0, 5000),
        'Transition\n(5K-15K)': (5000, 15000),
        'Plateau\n(15K-30K)': (15000, 30000),
        'Grokking\n(30K-100K)': (30000, 100000),
    }

    phase_names = []
    avg_improvements = []

    for phase_name, (start, end) in phases.items():
        mask = (np.array(steps) >= start) & (np.array(steps) < end)
        if np.any(mask):
            phase_names.append(phase_name)
            avg_improvements.append(np.mean(np.array(improvements)[mask]) * 100)  # 转换为百分比

    colors = ['red' if x < 0 else 'green' for x in avg_improvements]
    bars = ax.barh(phase_names, avg_improvements, color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Average Accuracy Improvement (%)', fontsize=12)
    ax.set_title('Intervention Effect by Training Phase', fontsize=12, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    # 在柱子上添加数值标签
    for bar, val in zip(bars, avg_improvements):
        x_pos = bar.get_width() + 0.5 if val >= 0 else bar.get_width() - 0.5
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}%', va='center', ha='left' if val >= 0 else 'right',
                fontsize=10, fontweight='bold')

    # 添加 Grokking 区域标注到子图 1 和 2
    for ax in [axes[0, 0], axes[0, 1]]:
        ax.axvspan(30000, max(steps), alpha=0.1, color='yellow')

    plt.suptitle('Grokking: Phase Intervention Experiment\\n'
                 'Hypothesis: Model has learned frequency magnitudes but lacks proper phase alignment\\n'
                 'Ideal Phase: $\phi_{{ideal}}[k, j] = 2\\pi k j / p$ (Linear phase distribution)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'phase_intervention.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file}')

    output_file_pdf = os.path.join(output_dir, 'phase_intervention.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file_pdf}')

    plt.close()


def main():
    phase_intervention_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/phase_intervention.csv"
    output_dir = "/root/data1/zjj/Grokking_Formulation/experiments/figures/2D_DFT"

    print("=" * 60)
    print("绘制相位干预实验结果")
    print("=" * 60)
    print(f"相位干预数据文件: {phase_intervention_file}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    print("\n加载数据...")
    data = load_data(phase_intervention_file)
    print(f"数据点数: {len(data)}")

    print("\n生成图形...")
    plot_phase_intervention(data, output_dir)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
