#!/usr/bin/env python3
"""
绘制频域对齐随训练步数的变化

使用 f_alginment.csv 和 metric.csv 绘制双 y 轴图：
- 左 y 轴：训练和测试准确率
- 右 y 轴：平均频域对齐（余弦相似度）
- x 轴采用对数刻度
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_data(f_alginment_file, metric_file):
    """加载 f_alginment 和 metric 数据"""
    # 加载 f_alginment 数据
    f_alginment_data = []
    with open(f_alginment_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            f_alginment_data.append(row)

    # 加载 metric 数据
    metric_data = []
    with open(metric_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_data.append(row)

    return f_alginment_data, metric_data


def plot_f_alginment(f_alginment_data, metric_data, output_dir):
    """绘制频域对齐图"""
    # 提取数据
    steps = [int(row['step']) for row in f_alginment_data]
    train_accs = [float(row['train_acc']) for row in f_alginment_data]
    test_accs = [float(row['test_acc']) for row in f_alginment_data]
    freq_alignments = [float(row['mean_freq_alignment']) for row in f_alginment_data]

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

    # 创建第二个 y 轴用于频域对齐
    ax2 = ax1.twinx()
    ax2.set_ylabel('Frequency Alignment (Cosine Similarity)', fontsize=12, color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # 绘制频域对齐曲线
    line3, = ax2.plot(steps, freq_alignments, 'g-', label='Mean Freq Alignment',
                       linewidth=2, alpha=0.8)

    # 添加第二个 y 轴的图例
    ax2.legend([line3], [line3.get_label()], loc='upper right')

    # 设置标题
    plt.title('Grokking: Accuracy & Frequency Alignment\n'
                 'Cosine similarity between W_E and W_U^T in frequency domain',
                 fontsize=14, fontweight='bold')

    # 添加 Grokking 区域标注
    # 找到测试准确率开始快速上升的点（约 30000 步）
    grokking_start = 30000
    ax1.axvspan(grokking_start, steps[-1], alpha=0.1, color='yellow', label='Grokking Phase')
    ax1.legend(loc='upper left')

    # 添加说明文字
    textstr = 'Freq Alignment: How well input & output\n' \
               'embeddings align in frequency space'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax1.text(0.02, 0.02, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)

    plt.tight_layout()

    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'f_alginment.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print('图表已保存至: {}'.format(output_file))

    # 同时保存 PDF 版本
    output_file_pdf = os.path.join(output_dir, 'f_alginment.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print('图表已保存至: {}'.format(output_file_pdf))

    plt.close()


def main():
    """主函数"""
    # 配置参数
    f_alginment_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/f_alginment.csv"
    metric_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/metric.csv"
    output_dir = "/root/data1/zjj/Grokking_Formulation/experiments/figures"

    print("=" * 60)
    print("绘制频域对齐变化图")
    print("=" * 60)
    print("频域对齐数据文件: {}".format(f_alginment_file))
    print("Metric 数据文件: {}".format(metric_file))
    print("输出目录: {}".format(output_dir))
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    f_alginment_data, metric_data = load_data(f_alginment_file, metric_file)

    print("频域对齐数据点数: {}".format(len(f_alginment_data)))
    print("Metric 数据点数: {}".format(len(metric_data)))

    # 绘制图形
    print("\n生成图形...")
    plot_f_alginment(f_alginment_data, metric_data, output_dir)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
