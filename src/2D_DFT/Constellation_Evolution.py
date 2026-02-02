#!/usr/bin/env python3
"""
星座图演化分析：可视化符号在复平面上的组织方式

对于选定的关键频率 k*，提取其在复平面上的表示：
1. 对 W_e 的每一列进行 DFT
2. 提取频率 k* 处的复数值
3. 绘制复平面散点图，颜色表示 token_id

预期：从初期的杂乱分布演化到后期的圆周均匀分布
"""

import os
import csv
import torch
import numpy as np
from scipy import signal


def find_dominant_frequency(p, d_model):
    """
    选择关键频率 k*

    策略：选择 k=1（基频），这是模运算最核心的频率
    对于 x+y (mod p) 问题，基频 k=1 对应线性结构

    Args:
        p: 模数
        d_model: 嵌入维度

    Returns:
        k_star: 选定的关键频率索引
    """
    # 对于模运算问题，k=1 是最关键的频率
    # 它对应于加法群 Z_p 上的线性表示
    return 1


def extract_constellation_data(W_e, k_star, p):
    """
    提取单个 checkpoint 的星座数据

    对 W_e 进行列向 DFT（沿 token 维度），然后提取频率 k* 处的值。
    由于 DFT 后是 (p, d_model) 的复数矩阵，我们需要为每个 token
    提取一个单一的复数值。

    策略：对所有特征维度取平均，得到 (p,) 的复数向量

    Args:
        W_e: (vocab_size, d_model) 嵌入矩阵
        k_star: 关键频率索引
        p: 模数

    Returns:
        constellation_data: List of (token_id, real, imag)
    """
    # 只取前 p 行（数字 token 0 到 p-1）
    W_e_p = W_e[:p, :]  # (p, d_model)

    # 对每列进行 DFT（沿 token 维度）
    W_e_dft = np.fft.fft(W_e_p, axis=0)  # (p, d_model)

    # 提取频率 k* 处的行：形状 (d_model,)
    # 这是所有特征在频率 k* 处的值
    freq_row = W_e_dft[k_star, :]  # (d_model,)

    # 对特征维度取平均，得到该频率的一个代表性复数值
    constellation_value = np.mean(freq_row)  # 标量复数

    # 为了展示 p 个 token 在复平面上的分布，我们需要为每个 token
    # 获取其在频率 k* 处的复数值
    # 由于 DFT 是沿 token 维度进行的，每个 token 对应 DFT 结果的"位置"
    # 而不是"值"

    # 另一种理解：我们想看每个 token 的嵌入在频域的表示
    # 可以对每行（每个 token）做 DFT，然后看 k* 频率的值
    constellation_per_token = []
    for token_id in range(p):
        token_embedding = W_e_p[token_id, :]  # (d_model,)
        token_dft = np.fft.fft(token_embedding)  # (d_model,)
        # 取 k* 频率（如果 k* < d_model）
        if k_star < len(token_dft):
            value = token_dft[k_star]
        else:
            value = 0j
        constellation_per_token.append(value)

    data = []
    for token_id in range(p):
        real = float(np.real(constellation_per_token[token_id]))
        imag = float(np.imag(constellation_per_token[token_id]))
        data.append((token_id, real, imag))

    return data


def extract_key_checkpoints(checkpoint_dir, key_steps):
    """
    提取关键时间点的星座数据

    Args:
        checkpoint_dir: checkpoint 目录
        key_steps: 关键步数列表

    Returns:
        all_data: List of (step, token_id, real, imag)
    """
    all_data = []

    for step in key_steps:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
        if not os.path.exists(checkpoint_path):
            print(f"警告: Checkpoint {checkpoint_path} 不存在，跳过")
            continue

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            W_e = checkpoint['model_state_dict']['embedding.weight'].numpy()

            p = 97
            k_star = find_dominant_frequency(p, W_e.shape[1])

            constellation_data = extract_constellation_data(W_e, k_star, p)

            for token_id, real, imag in constellation_data:
                all_data.append({
                    'step': step,
                    'token_id': token_id,
                    'real': real,
                    'imag': imag,
                })

            print(f"Step {step}: 提取了 {len(constellation_data)} 个 token 的星座数据")

        except Exception as e:
            print(f"处理 {checkpoint_path} 时出错: {e}")
            import traceback
            traceback.print_exc()

    return all_data


def analyze_spectral_energy(checkpoint_dir):
    """
    分析频域能量分布，帮助选择关键频率

    Args:
        checkpoint_dir: checkpoint 目录

    Returns:
        推荐的关键频率
    """
    # 使用训练后期的 checkpoint
    late_checkpoint = os.path.join(checkpoint_dir, "checkpoint_step_50000.pt")
    if not os.path.exists(late_checkpoint):
        late_checkpoint = os.path.join(checkpoint_dir, "checkpoint_step_100000.pt")

    if not os.path.exists(late_checkpoint):
        print("无法找到后期 checkpoint，使用默认 k=1")
        return 1

    try:
        checkpoint = torch.load(late_checkpoint, map_location='cpu')
        W_e = checkpoint['model_state_dict']['embedding.weight'].numpy()

        p = 97
        W_e_p = W_e[:p, :]

        # 计算频谱能量
        W_e_dft = np.fft.fft(W_e_p, axis=0)
        power_spectrum = np.abs(W_e_dft) ** 2

        # 对每个频率计算平均功率
        avg_power_per_freq = power_spectrum.mean(axis=1)

        # 排除 DC 分量 (k=0)，找到功率最高的频率
        avg_power_per_freq[0] = 0
        dominant_freq = np.argmax(avg_power_per_freq)

        print(f"频域能量分析结果:")
        print(f"  k=0 (DC): {np.abs(W_e_dft[0, :]).mean():.6f}")
        print(f"  k=1: {np.abs(W_e_dft[1, :]).mean():.6f}")
        print(f"  k={dominant_freq} (dominant): {np.abs(W_e_dft[dominant_freq, :]).mean():.6f}")

        # 对于模运算，k=1 通常是最有意义的
        return 1  # 返回基频

    except Exception as e:
        print(f"频谱分析出错: {e}")
        return 1


def main():
    checkpoint_dir = "/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints"
    output_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/constellation_data.csv"

    print("=" * 60)
    print("星座图演化分析：符号在复平面上的组织")
    print("=" * 60)
    print(f"Checkpoint 目录: {checkpoint_dir}")
    print(f"输出文件: {output_file}")

    # 定义 4 个关键时间点
    key_steps = [
        0,          # 初始化
        5000,       # 过拟合平台期（训练准确率 100%，测试准确率低）
        30000,      # Grokking 突变点附近
        99900,      # 收敛点（最后的 checkpoint）
    ]

    print(f"\n关键时间点: {key_steps}")
    print("  - Step 0: 初始化")
    print("  - Step 5000: 过拟合平台期")
    print("  - Step 30000: Grokking 突变点")
    print("  - Step 99900: 收敛点")

    # 分析频域能量
    print("\n" + "=" * 60)
    print("频域能量分析")
    print("=" * 60)
    k_star = analyze_spectral_energy(checkpoint_dir)
    print(f"\n选择的关键频率: k* = {k_star}")

    # 提取星座数据
    print("\n" + "=" * 60)
    print("提取星座数据")
    print("=" * 60)
    all_data = extract_key_checkpoints(checkpoint_dir, key_steps)

    # 保存数据
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', newline='') as f:
        fieldnames = ['step', 'token_id', 'real', 'imag']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    print(f"\n数据已保存至: {output_file}")
    print(f"共保存 {len(all_data)} 条记录")

    # 统计信息
    print("\n" + "=" * 60)
    print("数据统计")
    print("=" * 60)
    for step in key_steps:
        step_data = [d for d in all_data if d['step'] == step]
        if step_data:
            reals = [d['real'] for d in step_data]
            imags = [d['imag'] for d in step_data]
            magnitudes = [np.sqrt(r**2 + i**2) for r, i in zip(reals, imags)]

            print(f"\nStep {step}:")
            print(f"  Token 数量: {len(step_data)}")
            print(f"  实部范围: [{min(reals):.4f}, {max(reals):.4f}]")
            print(f"  虚部范围: [{min(imags):.4f}, {max(imags):.4f}]")
            print(f"  幅度范围: [{min(magnitudes):.4f}, {max(magnitudes):.4f}]")
            print(f"  平均幅度: {np.mean(magnitudes):.4f}")
            print(f"  幅度标准差: {np.std(magnitudes):.4f}")

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
