#!/usr/bin/env python3
"""
绘制星座图演化：符号在复平面上的组织方式

使用 constellation_data.csv 绘制 4 个子图（2x2），展示不同训练阶段
符号在复平面上的分布模式。
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_constellation_data(constellation_file):
    """加载星座数据"""
    data = {}
    with open(constellation_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row['step'])
            if step not in data:
                data[step] = {'token_id': [], 'real': [], 'imag': []}
            data[step]['token_id'].append(int(row['token_id']))
            data[step]['real'].append(float(row['real']))
            data[step]['imag'].append(float(row['imag']))
    return data


def plot_constellation_evolution(constellation_data, output_dir):
    """绘制星座图演化"""
    # 定义关键时间点及其标签
    key_steps_info = {
        0: ("Initialization", "初始化"),
        5000: ("Memorization Plateau", "过拟合平台期"),
        30000: ("Grokking Transition", "Grokking 转换点"),
        99900: ("Convergence", "收敛"),
    }

    # 创建图形
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    axes = [
        (fig.add_subplot(gs[0, 0]), 0),
        (fig.add_subplot(gs[0, 1]), 5000),
        (fig.add_subplot(gs[1, 0]), 30000),
        (fig.add_subplot(gs[1, 1]), 99900),
    ]

    # 全局数据用于统一刻度
    all_real = []
    all_imag = []
    for step, step_data in constellation_data.items():
        all_real.extend(step_data['real'])
        all_imag.extend(step_data['imag'])

    # 计算全局范围
    global_margin = 0.1
    real_min, real_max = min(all_real), max(all_real)
    imag_min, imag_max = min(all_imag), max(all_imag)
    real_range = real_max - real_min
    imag_range = imag_max - imag_min

    real_lim = (real_min - global_margin * real_range, real_max + global_margin * real_range)
    imag_lim = (imag_min - global_margin * imag_range, imag_max + global_margin * imag_range)

    # 绘制每个子图
    for ax, step in axes:
        if step not in constellation_data:
            ax.text(0.5, 0.5, f'No data for step {step}',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        step_data = constellation_data[step]
        token_ids = np.array(step_data['token_id'])
        reals = np.array(step_data['real'])
        imags = np.array(step_data['imag'])

        # 计算幅度
        magnitudes = np.sqrt(reals**2 + imags**2)

        # 使用 hsv colormap，颜色按 token_id 排序
        scatter = ax.scatter(reals, imags, c=token_ids, cmap='hsv',
                           s=60, alpha=0.8, edgecolors='black', linewidth=0.5)

        # 添加圆周参考线（使用平均幅度）
        avg_magnitude = np.mean(magnitudes)
        circle = plt.Circle((0, 0), avg_magnitude, fill=False,
                          color='gray', linestyle='--', alpha=0.3, linewidth=2)
        ax.add_patch(circle)

        # 添加坐标轴
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)

        # 设置标题和标签
        en_title, zh_title = key_steps_info.get(step, (f"Step {step}", f"步数 {step}"))
        ax.set_title(f'{en_title}\n({zh_title})', fontsize=13, fontweight='bold', pad=10)

        ax.set_xlabel('Real Part', fontsize=11)
        ax.set_ylabel('Imaginary Part', fontsize=11)

        # 统一坐标轴范围
        ax.set_xlim(real_lim)
        ax.set_ylim(imag_lim)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # 添加统计信息
        info_text = f'|z| mean: {avg_magnitude:.3f}\n'
        info_text += f'|z| std: {np.std(magnitudes):.3f}\n'
        info_text += f'Tokens: {len(token_ids)}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 标注几个关键 token
        key_tokens = [0, 24, 48, 72, 96]  # 均匀分布的 token
        for token in key_tokens:
            if token < len(token_ids):
                idx = np.where(token_ids == token)[0]
                if len(idx) > 0:
                    i = idx[0]
                    ax.annotate(f'token {token}',
                              (reals[i], imags[i]),
                              fontsize=8, alpha=0.7,
                              xytext=(5, 5), textcoords='offset points')

    # 添加总标题
    fig.suptitle('Grokking: Constellation Evolution in Complex Plane\n'
                f'Symbol Organization at Frequency $k^*=1$ (Fundamental Frequency)\\n'
                'Color gradient represents token_id (0 to p-1), showing sequential structure',
                fontsize=15, fontweight='bold', y=0.98)

    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Token ID', fontsize=12)

    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'constellation_evolution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file}')

    output_file_pdf = os.path.join(output_dir, 'constellation_evolution.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file_pdf}')

    plt.close()


def plot_phase_analysis(constellation_data, output_dir):
    """绘制相位分析图"""
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    key_steps_info = {
        0: ("Initialization", "初始化"),
        5000: ("Memorization Plateau", "过拟合平台期"),
        30000: ("Grokking Transition", "Grokking 转换点"),
        99900: ("Convergence", "收敛"),
    }

    step_list = [0, 5000, 30000, 99900]

    for idx, step in enumerate(step_list):
        ax = axes[idx // 2, idx % 2]

        if step not in constellation_data:
            continue

        step_data = constellation_data[step]
        token_ids = np.array(step_data['token_id'])
        reals = np.array(step_data['real'])
        imags = np.array(step_data['imag'])

        # 计算相位
        phases = np.arctan2(imags, reals)  # [-pi, pi]
        magnitudes = np.sqrt(reals**2 + imags**2)

        # 按 token_id 排序
        sort_idx = np.argsort(token_ids)
        sorted_tokens = token_ids[sort_idx]
        sorted_phases = phases[sort_idx]

        # 绘制相位 vs token_id
        scatter = ax.scatter(sorted_tokens, sorted_phases,
                           c=sorted_tokens, cmap='hsv',
                           s=40, alpha=0.7, edgecolors='none')

        # 添加理想线性参考线（均匀分布在圆周上）
        ideal_phases = 2 * np.pi * sorted_tokens / 97 - np.pi  # 归一化到 [-pi, pi]
        ax.plot(sorted_tokens, ideal_phases, 'r--', linewidth=2,
               alpha=0.5, label='Ideal (uniform on circle)')

        ax.set_xlabel('Token ID', fontsize=11)
        ax.set_ylabel('Phase (radians)', fontsize=11)
        ax.set_title(f'{key_steps_info[step][0]} ({key_steps_info[step][1]})',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # 添加统计信息
        # 计算相位线性度
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(sorted_tokens, sorted_phases)
        r_squared = r_value ** 2

        info_text = f'R² (linearity): {r_squared:.4f}\n'
        info_text += f'Phase range: [{np.min(sorted_phases):.2f}, {np.max(sorted_phases):.2f}]'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))

    plt.suptitle('Grokking: Phase vs Token ID Relationship\\n'
                'Linear phase distribution indicates uniform circular arrangement',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存图形
    output_file = os.path.join(output_dir, 'phase_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'图表已保存至: {output_file}')

    output_file_pdf = os.path.join(output_dir, 'phase_analysis.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f'图表已保存至: {output_file_pdf}')

    plt.close()


def main():
    constellation_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/constellation_data.csv"
    output_dir = "/root/data1/zjj/Grokking_Formulation/experiments/figures/2D_DFT"

    print("=" * 60)
    print("绘制星座图演化")
    print("=" * 60)
    print(f"星座数据文件: {constellation_file}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    print("\n加载数据...")
    constellation_data = load_constellation_data(constellation_file)
    print(f"加载了 {len(constellation_data)} 个时间点的数据")

    print("\n生成星座图演化...")
    plot_constellation_evolution(constellation_data, output_dir)

    print("\n生成相位分析图...")
    plot_phase_analysis(constellation_data, output_dir)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
