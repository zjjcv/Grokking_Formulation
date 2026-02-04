#!/usr/bin/env python3
"""
绘制电路竞争随训练步数的变化（记忆 vs 算法）

使用 circuit_Competition.csv 和 metric.csv 绘图：
- 展示记忆电路与算法电路的此消彼长
- 寻找 Crossover 点（算法超过记忆）
- x 轴采用对数刻度
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_data(circuit_competition_file, metric_file):
    """加载电路竞争和 metric 数据"""
    circuit_competition_data = []
    with open(circuit_competition_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            circuit_competition_data.append(row)

    metric_data = []
    with open(metric_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_data.append(row)

    return circuit_competition_data, metric_data


def find_crossover_point(data):
    """
    寻找 Crossover 点（算法能量超过记忆能量）

    Args:
        data: 电路竞争数据

    Returns:
        crossover_step: Crossover 发生的步数（如果存在）
        crossover_idx: Crossover 发生的索引
    """
    for i in range(1, len(data)):
        prev_memo = float(data[i-1]['W_U_memo_energy'])
        prev_fourier = float(data[i-1]['W_U_fourier_energy'])
        curr_memo = float(data[i]['W_U_memo_energy'])
        curr_fourier = float(data[i]['W_U_fourier_energy'])

        # 检测交叉
        if prev_memo > prev_fourier and curr_fourier > curr_memo:
            return int(data[i]['step']), i
    return None, None


def plot_circuit_competition(circuit_competition_data, metric_data, output_dir):
    """绘制电路竞争图"""
    steps = [int(row['step']) for row in circuit_competition_data]
    train_accs = [float(row['train_acc']) for row in circuit_competition_data]
    test_accs = [float(row['test_acc']) for row in circuit_competition_data]

    # W_E 数据
    memo_energies_E = [float(row['W_E_memo_energy']) for row in circuit_competition_data]
    fourier_energies_E = [float(row['W_E_fourier_energy']) for row in circuit_competition_data]
    residual_energies_E = [float(row['W_E_residual_energy']) for row in circuit_competition_data]

    # W_U 数据
    memo_energies_U = [float(row['W_U_memo_energy']) for row in circuit_competition_data]
    fourier_energies_U = [float(row['W_U_fourier_energy']) for row in circuit_competition_data]
    residual_energies_U = [float(row['W_U_residual_energy']) for row in circuit_competition_data]
    ratios_U = [float(row['W_U_competition_ratio']) for row in circuit_competition_data]

    # 寻找 Crossover 点
    crossover_step, crossover_idx = find_crossover_point(circuit_competition_data)

    # ========== 图1: 主图 - 堆叠面积图展示此消彼长 ==========
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # W_E 子图
    ax = axes[0]
    ax.set_ylabel('Projection Energy (W_E)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    ax.fill_between(steps, 0, memo_energies_E, alpha=0.7, color='red', label='Memory Circuit')
    ax.fill_between(steps, memo_energies_E,
                    np.array(memo_energies_E) + np.array(fourier_energies_E),
                    alpha=0.7, color='blue', label='Algorithm Circuit')
    ax.fill_between(steps, np.array(memo_energies_E) + np.array(fourier_energies_E),
                    np.array(memo_energies_E) + np.array(fourier_energies_E) + np.array(residual_energies_E),
                    alpha=0.3, color='gray', label='Residual')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title('Input Embedding (W_E): Memory vs Algorithm', fontsize=12, fontweight='bold')

    ax_r = ax.twinx()
    ax_r.plot(steps, test_accs, 'g:', linewidth=1.5, alpha=0.5, label='Test Acc')
    ax_r.set_ylabel('Accuracy', fontsize=11)

    # W_U 子图
    ax = axes[1]
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Projection Energy (W_U)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # 如果找到 Crossover 点，标记它
    if crossover_step is not None:
        ax.axvline(x=crossover_step, color='purple', linestyle='--', linewidth=2,
                  label=f'Crossover (Step {crossover_step})', alpha=0.8)

    ax.fill_between(steps, 0, memo_energies_U, alpha=0.7, color='red', label='Memory Circuit')
    ax.fill_between(steps, memo_energies_U,
                    np.array(memo_energies_U) + np.array(fourier_energies_U),
                    alpha=0.7, color='blue', label='Algorithm Circuit')
    ax.fill_between(steps, np.array(memo_energies_U) + np.array(fourier_energies_U),
                    np.array(memo_energies_U) + np.array(fourier_energies_U) + np.array(residual_energies_U),
                    alpha=0.3, color='gray', label='Residual')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title('Output Embedding (W_U): Memory vs Algorithm (Crossover)', fontsize=12, fontweight='bold')

    ax_r = ax.twinx()
    ax_r.plot(steps, test_accs, 'g:', linewidth=1.5, alpha=0.5, label='Test Acc')
    ax_r.set_ylabel('Accuracy', fontsize=11)

    # 添加 Grokking 区域标注
    for ax in axes:
        grokking_start = 30000
        ax.axvspan(grokking_start, max(steps), alpha=0.1, color='yellow')

    if crossover_step:
        title_suffix = f'\nCrossover at Step {crossover_step}'
    else:
        title_suffix = ''

    plt.suptitle(f'Grokking: Circuit Competition (Memory vs Algorithm){title_suffix}\n'
                 'Red: Memory Circuit | Blue: Algorithm Circuit | Crossover: Algorithm > Memory',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'circuit_Competition.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file}')

    output_file_pdf = os.path.join(output_dir, 'circuit_Competition.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file_pdf}')

    plt.close()

    # ========== 图2: 竞争比率图 ==========
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    line1, = ax.plot(steps, train_accs, 'b-', label='Train Acc', linewidth=1.5, alpha=0.5)
    line2, = ax.plot(steps, test_accs, 'r-', label='Test Acc', linewidth=2, alpha=0.7)
    ax.legend(loc='upper left', fontsize=10)

    ax2 = ax.twinx()
    ax2.set_ylabel('Algorithm / Memory Ratio', fontsize=12, color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    line3, = ax2.plot(steps, ratios_U, 'orange', linewidth=2, label='W_U Ratio', alpha=0.8)
    ax2.legend([line3], [line3.get_label()], loc='center right', fontsize=10)

    # 添加参考线
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Balance (ratio=1)')
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

    # 标记 Crossover 点
    if crossover_step:
        ax2.axvline(x=crossover_step, color='purple', linestyle='--', linewidth=2, alpha=0.8)

    # 填充区域
    ax2.fill_between(steps, 0, ratios_U, where=np.array(ratios_U)<1,
                     alpha=0.3, color='red', label='Memory Dominant')
    ax2.fill_between(steps, 0, ratios_U, where=np.array(ratios_U)>=1,
                     alpha=0.3, color='blue', label='Algorithm Dominant')

    plt.title('Grokking: Circuit Competition Ratio\n'
              'Ratio > 1: Algorithm Dominant | Ratio < 1: Memory Dominant',
              fontsize=14, fontweight='bold')

    # 添加 Grokking 区域
    grokking_start = 30000
    ax.axvspan(grokking_start, max(steps), alpha=0.1, color='yellow', label='Grokking Phase')

    textstr = 'Competition Ratio = Algorithm Energy / Memory Energy\n' \
               'Crossover: Algorithm circuit overtakes memory circuit'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)

    plt.tight_layout()

    output_file2 = os.path.join(output_dir, 'circuit_Competition_ratio.png')
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file2}')

    output_file2_pdf = os.path.join(output_dir, 'circuit_Competition_ratio.pdf')
    plt.savefig(output_file2_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file2_pdf}')

    plt.close()

    # ========== 图3: 详细多子图 ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # W_U 能量分量
    ax = axes[0, 0]
    ax.plot(steps, memo_energies_U, 'r-', linewidth=2, label='Memory')
    ax.plot(steps, fourier_energies_U, 'b-', linewidth=2, label='Algorithm')
    if crossover_step:
        ax.axvline(x=crossover_step, color='purple', linestyle='--', linewidth=2, alpha=0.8)
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Energy', fontsize=11)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('W_U: Memory vs Algorithm Energy', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    # 竞争比率
    ax = axes[0, 1]
    ax.plot(steps, ratios_U, 'orange', linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    if crossover_step:
        ax.axvline(x=crossover_step, color='purple', linestyle='--', linewidth=2, alpha=0.8)
    ax.fill_between(steps, 0, ratios_U, where=np.array(ratios_U)<1,
                    alpha=0.3, color='red')
    ax.fill_between(steps, 0, ratios_U, where=np.array(ratios_U)>=1,
                    alpha=0.3, color='blue')
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Algorithm / Memory Ratio', fontsize=11)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('Competition Ratio (Crossover at ratio=1)', fontsize=12, fontweight='bold')

    # 能量差值
    ax = axes[1, 0]
    energy_diff = np.array(fourier_energies_U) - np.array(memo_energies_U)
    ax.plot(steps, energy_diff, 'purple', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    if crossover_step:
        ax.axvline(x=crossover_step, color='purple', linestyle='--', linewidth=2, alpha=0.8)
    ax.fill_between(steps, 0, energy_diff, where=np.array(energy_diff)<0,
                    alpha=0.3, color='red', label='Memory > Algorithm')
    ax.fill_between(steps, 0, energy_diff, where=np.array(energy_diff)>=0,
                    alpha=0.3, color='blue', label='Algorithm > Memory')
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Algorithm - Memory Energy', fontsize=11)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('Energy Difference (Crossover at diff=0)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    # 累积能量
    ax = axes[1, 1]
    cumulative_memo = np.cumsum(memo_energies_U)
    cumulative_fourier = np.cumsum(fourier_energies_U)
    ax.plot(steps, cumulative_memo, 'r--', linewidth=1.5, alpha=0.7, label='Memory (cumulative)')
    ax.plot(steps, cumulative_fourier, 'b-', linewidth=2, alpha=0.8, label='Algorithm (cumulative)')
    if crossover_step:
        ax.axvline(x=crossover_step, color='purple', linestyle='--', linewidth=2, alpha=0.8)
    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_ylabel('Cumulative Energy', fontsize=11)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_title('Cumulative Energy Over Training', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)

    # 添加 Grokking 区域
    for ax in axes.flat:
        grokking_start = 30000
        ax.axvspan(grokking_start, max(steps), alpha=0.1, color='yellow')

    plt.suptitle('Circuit Competition: Detailed Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file3 = os.path.join(output_dir, 'circuit_Competition_detailed.png')
    plt.savefig(output_file3, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file3}')

    output_file3_pdf = os.path.join(output_dir, 'circuit_Competition_detailed.pdf')
    plt.savefig(output_file3_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file3_pdf}')

    plt.close()


def main():
    circuit_competition_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/circuit_Competition.csv"
    metric_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/metric.csv"
    output_dir = "/root/data1/zjj/Grokking_Formulation/experiments/figures"

    print("=" * 60)
    print("绘制电路竞争变化图（记忆 vs 算法）")
    print("=" * 60)
    print(f"电路竞争数据文件: {circuit_competition_file}")
    print(f"Metric 数据文件: {metric_file}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    print("\n加载数据...")
    circuit_competition_data, metric_data = load_data(circuit_competition_file, metric_file)

    print(f"电路竞争数据点数: {len(circuit_competition_data)}")
    print(f"Metric 数据点数: {len(metric_data)}")

    print("\n生成图形...")
    plot_circuit_competition(circuit_competition_data, metric_data, output_dir)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
