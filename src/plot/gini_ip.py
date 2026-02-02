#!/usr/bin/env python3
"""
绘制基尼系数和逆参与率随训练步数的变化（频域分析）

使用 gini_ip.csv 和 metric.csv 绘制三 y 轴图：
- 左 y 轴：训练和测试准确率
- 右 y 轴：基尼系数和逆参与率（在频域计算的值）
- x 轴采用对数刻度
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_data(gini_ip_file, metric_file):
    """加载 gini_ip 和 metric 数据"""
    # 加载 gini_ip 数据
    gini_ip_data = []
    with open(gini_ip_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gini_ip_data.append(row)

    # 加载 metric 数据
    metric_data = []
    with open(metric_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_data.append(row)

    return gini_ip_data, metric_data


def plot_gini_ip(gini_ip_data, metric_data, output_dir):
    """绘制基尼系数和逆参与率图（频域）"""
    # 提取数据
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
    ax2.set_ylabel('Gini Coefficient / IPR (Frequency Domain)', fontsize=12, color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # 绘制基尼系数
    line3, = ax2.plot(steps, ginis, 'g-', label='Gini Coefficient (Freq)', linewidth=2, alpha=0.8)
    # 绘制逆参与率（归一化到 [0, 1] 范围以便显示）
    iprs_normalized = [(ipr - min(iprs)) / (max(iprs) - min(iprs)) for ipr in iprs]
    line4, = ax2.plot(steps, iprs_normalized, 'm--', label='IPR (Freq, normalized)', linewidth=2, alpha=0.8)

    # 添加第二个 y 轴的图例
    lines = [line3, line4]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')

    # 设置标题
    plt.title('Grokking: Accuracy, Gini Coefficient & Inverse Participation Ratio (Frequency Domain)\n'
                 'Gini: inequality of DFT magnitude | IPR: effective dimensionality of frequency spectrum',
                 fontsize=14, fontweight='bold')

    # 添加 Grokking 区域标注
    # 找到测试准确率开始快速上升的点（约 30000 步）
    grokking_start = 30000
    ax1.axvspan(grokking_start, steps[-1], alpha=0.1, color='yellow', label='Grokking Phase')
    ax1.legend(loc='upper left')

    # 添加说明文字
    textstr = 'Metrics computed on DFT magnitude of embedding\n' \
               'Gini Coeff: Lower = more unequal freq distribution\n' \
               'IPR: Higher = more diverse frequency components'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax1.text(0.02, 0.02, textstr, transform=ax1.transAxes, fontsize=8,
            verticalalignment='bottom', bbox=props)

    plt.tight_layout()

    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'gini_ip.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print('图表已保存至: {}'.format(output_file))

    # 同时保存 PDF 版本
    output_file_pdf = os.path.join(output_dir, 'gini_ip.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print('图表已保存至: {}'.format(output_file_pdf))

    plt.close()


def main():
    """主函数"""
    # 配置参数
    gini_ip_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/gini_ip.csv"
    metric_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/metric.csv"
    output_dir = "/root/data1/zjj/Grokking_Formulation/experiments/figures"

    print("=" * 60)
    print("绘制基尼系数和逆参与率变化图（频域分析）")
    print("=" * 60)
    print("Gini/IP 数据文件: {}".format(gini_ip_file))
    print("Metric 数据文件: {}".format(metric_file))
    print("输出目录: {}".format(output_dir))
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    gini_ip_data, metric_data = load_data(gini_ip_file, metric_file)

    print("Gini/IP 数据点数: {}".format(len(gini_ip_data)))
    print("Metric 数据点数: {}".format(len(metric_data)))

    # 绘制图形
    print("\n生成图形...")
    plot_gini_ip(gini_ip_data, metric_data, output_dir)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
