#!/usr/bin/env python3
"""
绘制功率谱密度(PSD)随训练步数的变化

使用 psd.csv 和 metric.csv 绘制双 y 轴图：
- 左 y 轴：训练和测试准确率
- 右 y 轴：频率（0 到 p/2）
- PSD 使用热力图形式，颜色代表该频率下的平均能量
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm


def load_data(psd_file, metric_file):
    """加载 PSD 和 metric 数据"""
    # 加载 PSD 数据
    psd_data = []
    with open(psd_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            psd_data.append(row)

    # 加载 metric 数据
    metric_data = []
    with open(metric_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_data.append(row)

    return psd_data, metric_data


def parse_psd_data(psd_data, p):
    """解析 PSD 数据，提取频率列"""
    steps = []
    train_accs = []
    test_accs = []

    # 找出频率列 (freq_0, freq_1, ..., freq_{p//2})
    freq_cols = ['freq_{}'.format(i) for i in range(p // 2 + 1)]

    # 收集 PSD 数据
    psd_values = {}
    for i in range(p // 2 + 1):
        psd_values['freq_{}'.format(i)] = []

    for row in psd_data:
        steps.append(int(row['step']))
        train_accs.append(float(row['train_acc']))
        test_accs.append(float(row['test_acc']))

        for freq_col in freq_cols:
            if freq_col in row:
                psd_values[freq_col].append(float(row[freq_col]))
            else:
                psd_values[freq_col].append(0.0)

    return steps, train_accs, test_accs, psd_values


def plot_psd_heatmap(steps, train_accs, test_accs, psd_values, p, output_dir):
    """绘制 PSD 热力图"""
    # 准备频率和 PSD 数据
    frequencies = list(range(p // 2 + 1))
    freq_cols = ['freq_{}'.format(i) for i in frequencies]

    # 构建 PSD 矩阵 (steps x frequencies)
    psd_matrix = np.zeros((len(steps), len(frequencies)))
    for i, freq_col in enumerate(freq_cols):
        psd_matrix[:, i] = psd_values[freq_col]

    # 创建图形
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # 绘制准确率曲线
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    line1, = ax1.plot(steps, train_accs, 'b-', label='Train Acc', linewidth=1.5, alpha=0.7)
    line2, = ax1.plot(steps, test_accs, 'r-', label='Test Acc', linewidth=1.5, alpha=0.7)
    ax1.legend(loc='upper left')

    # 创建第二个 y 轴用于频率
    ax2 = ax1.twinx()
    ax2.set_ylabel('Frequency', fontsize=12, color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.set_ylim(0, p // 2)

    # 创建 PSD 热力图
    # X 轴：步数，Y 轴：频率，颜色：PSD 值
    extent = [steps[0], steps[-1], 0, p // 2]

    im = ax1.imshow(psd_matrix.T,
                   aspect='auto',
                   origin='lower',
                   extent=extent,
                   cmap='viridis',
                   alpha=0.6,
                   interpolation='bilinear')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax1, pad=0.02)
    cbar.set_label('Normalized Power Spectral Density', fontsize=11)

    # 设置标题
    plt.title('Grokking: Accuracy & Power Spectral Density of Embedding\n'
                 'Heatmap shows PSD across frequencies (0 to p/2)',
                 fontsize=14, fontweight='bold')

    # 添加图例说明
    textstr = 'Heatmap Color: PSD Energy (Yellow=High, Purple=Low)\n' \
               'Frequency: 0 to p/2 (modular arithmetic relevant range)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax1.text(0.02, 0.02, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)

    plt.tight_layout()

    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'psd_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print('图表已保存至: {}'.format(output_file))

    # 同时保存 PDF 版本
    output_file_pdf = os.path.join(output_dir, 'psd_heatmap.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print('图表已保存至: {}'.format(output_file_pdf))

    plt.close()


def main():
    """主函数"""
    # 配置参数
    psd_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/psd.csv"
    metric_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/metric.csv"
    output_dir = "/root/data1/zjj/Grokking_Formulation/experiments/figures"
    p = 97  # 模数

    print("=" * 60)
    print("绘制功率谱密度热力图")
    print("=" * 60)
    print("PSD 数据文件: {}".format(psd_file))
    print("Metric 数据文件: {}".format(metric_file))
    print("输出目录: {}".format(output_dir))
    print("模数 p: {}".format(p))
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    psd_data, metric_data = load_data(psd_file, metric_file)

    print("PSD 数据点数: {}".format(len(psd_data)))
    print("Metric 数据点数: {}".format(len(metric_data)))

    # 解析 PSD 数据
    print("\n解析 PSD 数据...")
    steps, train_accs, test_accs, psd_values = parse_psd_data(psd_data, p)

    print("步数范围: {} 到 {}".format(steps[0], steps[-1]))
    print("频率范围: 0 到 {}".format(p // 2))

    # 绘制图形
    print("\n生成图形...")
    plot_psd_heatmap(steps, train_accs, test_accs, psd_values, p, output_dir)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
