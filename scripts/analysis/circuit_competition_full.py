#!/usr/bin/env python3
"""
提取 Transformer 权重，分析电路竞争（记忆 vs 算法）

定义两个子空间：
1. S_memo: 记忆子空间，由初始化阶段大梯度样本的梯度主成分定义
2. S_fourier: 算法子空间，由 DFT 矩阵的前 k 个低频分量定义

追踪模型权重在这两个子空间上的投影分量能量占比，
观察记忆电路与算法电路的此消彼长（Crossover）。
"""

import os
import csv
import torch
import numpy as np
from scipy.linalg import svd


def create_fourier_subspace(n, k=20):
    """
    创建傅里叶子空间（前 k 个低频分量）

    Args:
        n: 嵌入维度
        k: 使用的低频分量数量

    Returns:
        fourier_basis: (n, k) 傅里叶基矩阵（前 k 个低频）
    """
    basis = np.zeros((n, k), dtype=complex)
    for i in range(k):
        freq = i  # 低频分量
        basis[:, i] = np.exp(-2j * np.pi * freq * np.arange(n) / n) / np.sqrt(n)
    return basis


def compute_initial_gradients(model, data_samples, criterion):
    """
    计算初始化阶段模型在大梯度样本上的梯度

    Args:
        model: 模型
        data_samples: 数据样本
        criterion: 损失函数

    Returns:
        gradients: 各个参数的梯度
    """
    # 这是一个简化的实现
    # 实际上需要加载初始化模型并计算梯度
    # 这里我们使用一种近似方法：使用早期 checkpoint 的权重变化

    # 由于我们只有 checkpoint，无法直接计算初始梯度
    # 我们使用 step 0 和 step 100 的权重差作为近似
    return None


def identify_memo_subspace_from_early_checkpoints(checkpoint_dir, n_components=10):
    """
    从早期 checkpoint 识别记忆子空间

    使用训练早期（如前 1000 步）的权重变化主成分作为记忆方向

    Args:
        checkpoint_dir: checkpoint 目录
        n_components: 主成分数量

    Returns:
        memo_basis: (embed_dim, n_components) 记忆子空间基
    """
    # 加载早期 checkpoint 的权重
    weights_early = []
    steps_to_check = [0, 100, 200, 500, 1000]

    for step in steps_to_check:
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step}.pt')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            W_E = checkpoint['model_state_dict']['embedding.weight'].numpy()
            weights_early.append(W_E.flatten())

    weights_early = np.array(weights_early)  # (n_checkpoints, n_params)

    # 计算主成分分析（PCA）
    # 中心化
    mean_weights = np.mean(weights_early, axis=0)
    centered_weights = weights_early - mean_weights

    # SVD 分解获取主成分
    U, S, Vt = svd(centered_weights, full_matrices=False)

    # 前 n_components 个主成分方向
    # 需要重塑为原始形状 (vocab_size, embed_dim)
    embed_dim = 128  # 已知
    vocab_size = 98  # 已知

    # 主成分是参数空间的向量
    # 我们取前 n_components 个右奇异向量
    components = Vt[:n_components, :]  # (n_components, n_params)

    # 将每个主成分重塑为 (vocab_size, embed_dim)，然后取转置得到 (embed_dim, vocab_size)
    # 再取 SVD 得到 (embed_dim, n_components)
    memo_basis = np.zeros((embed_dim, n_components))

    for i in range(n_components):
        # 重塑主成分
        pc_reshaped = components[i].reshape(vocab_size, embed_dim)  # (vocab_size, embed_dim)
        # 对这个矩阵进行 SVD，取前 n_components 个左奇异向量作为基
        try:
            U_pc, S_pc, Vt_pc = svd(pc_reshaped, full_matrices=False)
            memo_basis[:, i] = U_pc[:, 0]  # 取第一主成分
        except:
            memo_basis[:, i] = np.random.randn(embed_dim)
            memo_basis[:, i] /= np.linalg.norm(memo_basis[:, i])

    # 正交化
    from scipy.linalg import qr
    memo_basis, _ = qr(memo_basis, mode='economic')

    return memo_basis


def create_fourier_subspace_dynamic(embedding_matrix, n_freqs=10):
    """
    创建动态傅里叶子空间：从当前权重的 DFT 中提取低频结构

    方法：
    1. 对嵌入矩阵进行 2D DFT
    2. 识别低频区域（左上角）
    3. 通过逆 DFT 重建对应的原始空间基向量
    4. 这些基向量代表"算法方向"——低频、结构化的表示

    Args:
        embedding_matrix: (vocab_size, embed_dim) 嵌入矩阵
        n_freqs: 使用的低频分量数量

    Returns:
        fourier_basis: (embed_dim, n_freqs) 算法子空间基矩阵
    """
    # 进行 2D DFT
    dft_result = np.fft.fft2(embedding_matrix)

    # 创建低频模板（只保留低频，其他置零）
    low_freq_mask = np.zeros_like(dft_result)
    vocab_size, embed_dim = embedding_matrix.shape

    # 保留左上角的低频区域
    freq_region = min(n_freqs, vocab_size // 2, embed_dim // 2)
    for i in range(freq_region):
        for j in range(freq_region):
            low_freq_mask[i, j] = 1.0
            # 对称的高频部分（DFT 的性质）
            if i > 0:
                low_freq_mask[-i, j] = 1.0
            if j > 0:
                low_freq_mask[i, -j] = 1.0

    # 应用低频模板
    low_freq_dft = dft_result * low_freq_mask

    # 逆 DFT 得到低频成分
    low_freq_component = np.fft.ifft2(low_freq_dft).real

    # 对低频成分进行 SVD，提取主要方向作为基
    U, S, Vt = svd(low_freq_component, full_matrices=False)

    # 取前 n_freqs 个右奇异向量作为基
    fourier_basis = Vt[:n_freqs, :].T  # (embed_dim, n_freqs)

    return fourier_basis


def identify_memo_subspace_simple(embedding_matrix, n_components=10):
    """
    记忆子空间：使用嵌入矩阵的主成分（整体方差方向）

    Args:
        embedding_matrix: (vocab_size, embed_dim) 嵌入矩阵
        n_components: 主成分数量

    Returns:
        memo_basis: (embed_dim, n_components) 记忆子空间基
    """
    # 对嵌入矩阵进行 SVD
    U, S, Vt = svd(embedding_matrix, full_matrices=False)

    # 取前 n_components 个右奇异向量作为记忆子空间基
    memo_basis = Vt[:n_components, :].T  # (embed_dim, n_components)

    return memo_basis


def compute_projection_energy_ratio(weight_matrix, subspace_basis):
    """
    计算权重矩阵在子空间上的投影能量占比

    Args:
        weight_matrix: (m, n) 权重矩阵
        subspace_basis: (n, k) 子空间基矩阵（k 个基向量）

    Returns:
        energy_ratio: 投影能量占比 [0, 1]
    """
    # 投影：P = W @ V @ V^H，其中 V 是基矩阵，V^H 是共轭转置
    V = subspace_basis
    V_H = V.conj().T

    # 计算投影
    projected = weight_matrix @ V @ V_H

    # 计算能量
    original_energy = np.sum(np.abs(weight_matrix) ** 2)
    projection_energy = np.sum(np.abs(projected) ** 2)

    ratio = projection_energy / (original_energy + 1e-10)

    return ratio


def compute_circuit_competition(checkpoint_dir, output_file, n_memo_components=10, n_fourier_freqs=20):
    """
    从所有 checkpoint 文件中计算电路竞争指标

    新的算法子空间定义：
    - 对每个 checkpoint 的权重进行 DFT
    - 提取低频成分
    - 对低频成分进行 SVD 得到算法子空间基

    这样算法子空间是动态的，能捕捉模型学到的低频结构

    Args:
        checkpoint_dir: checkpoint 文件目录
        output_file: 输出 CSV 文件路径
        n_memo_components: 记忆子空间主成分数
        n_fourier_freqs: 算法子空间低频分量数
    """
    # 获取所有 checkpoint 文件并按步数排序
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("checkpoint_step_") and f.endswith(".pt"):
            step = int(f.replace("checkpoint_step_", "").replace(".pt", ""))
            checkpoint_files.append((step, os.path.join(checkpoint_dir, f)))

    checkpoint_files.sort(key=lambda x: x[0])

    print(f"找到 {len(checkpoint_files)} 个 checkpoint 文件")

    # 使用第一个 checkpoint 定义记忆子空间（初始化状态）
    first_checkpoint = torch.load(checkpoint_files[0][1], map_location='cpu')
    W_E_init = first_checkpoint['model_state_dict']['embedding.weight'].numpy()
    W_U_init = first_checkpoint['model_state_dict']['output.weight'].numpy()

    # 定义记忆子空间（使用初始化嵌入的主成分）
    memo_basis_E = identify_memo_subspace_simple(W_E_init, n_components=n_memo_components)
    memo_basis_U = identify_memo_subspace_simple(W_U_init, n_components=n_memo_components)

    # 准备输出数据
    results = []

    # 处理每个 checkpoint
    for step, checkpoint_path in checkpoint_files:
        try:
            # 加载 checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 提取输入嵌入 W_E: (vocab_size, embed_dim) = (98, 128)
            W_E = checkpoint['model_state_dict']['embedding.weight'].numpy()

            # 提取输出权重 W_U: (p, embed_dim) = (97, 128)
            W_U = checkpoint['model_state_dict']['output.weight'].numpy()

            # === 动态定义算法子空间（对当前权重提取低频结构）===
            fourier_basis_E = create_fourier_subspace_dynamic(W_E, n_freqs=n_fourier_freqs)
            fourier_basis_U = create_fourier_subspace_dynamic(W_U, n_freqs=n_fourier_freqs)

            # 计算 W_E 在各子空间的投影能量占比
            memo_energy_E = compute_projection_energy_ratio(W_E, memo_basis_E)
            fourier_energy_E = compute_projection_energy_ratio(W_E, fourier_basis_E)

            # 计算 W_U 在各子空间的投影能量占比
            memo_energy_U = compute_projection_energy_ratio(W_U, memo_basis_U)
            fourier_energy_U = compute_projection_energy_ratio(W_U, fourier_basis_U)

            # 计算其他能量
            residual_energy_E = max(0, 1 - memo_energy_E - fourier_energy_E)
            residual_energy_U = max(0, 1 - memo_energy_U - fourier_energy_U)

            # 计算竞争比率（算法/记忆）
            competition_ratio_E = fourier_energy_E / (memo_energy_E + 1e-10)
            competition_ratio_U = fourier_energy_U / (memo_energy_U + 1e-10)

            result = {
                'step': step,
                'train_loss': checkpoint['train_loss'],
                'train_acc': checkpoint['train_acc'],
                'test_loss': checkpoint['test_loss'],
                'test_acc': checkpoint['test_acc'],
                # W_E 投影能量
                'W_E_memo_energy': float(memo_energy_E),
                'W_E_fourier_energy': float(fourier_energy_E),
                'W_E_residual_energy': float(residual_energy_E),
                'W_E_competition_ratio': float(competition_ratio_E),
                # W_U 投影能量
                'W_U_memo_energy': float(memo_energy_U),
                'W_U_fourier_energy': float(fourier_energy_U),
                'W_U_residual_energy': float(residual_energy_U),
                'W_U_competition_ratio': float(competition_ratio_U),
            }

            results.append(result)

            if step % 100 == 0:
                print(f"Step {step}: W_U_Memo={memo_energy_U:.4f}, W_U_Algo={fourier_energy_U:.4f}, "
                      f"Ratio={competition_ratio_U:.4f}")

        except Exception as e:
            print(f"处理 {checkpoint_path} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 保存到 CSV
    if results:
        fieldnames = list(results[0].keys())

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n数据已保存至: {output_file}")
        print(f"共保存 {len(results)} 个时间步的数据")

        # 打印统计信息
        memo_energies_U = [r['W_U_memo_energy'] for r in results]
        fourier_energies_U = [r['W_U_fourier_energy'] for r in results]
        ratios_U = [r['W_U_competition_ratio'] for r in results]

        print("\nW_U 电路竞争统计:")
        print(f"  记忆能量: 初始={memo_energies_U[0]:.4f}, 最终={memo_energies_U[-1]:.4f}")
        print(f"  算法能量: 初始={fourier_energies_U[0]:.4f}, 最终={fourier_energies_U[-1]:.4f}")
        print(f"  竞争比率: 初始={ratios_U[0]:.4f}, 最终={ratios_U[-1]:.4f}")

        # 寻找 Crossover 点（算法能量超过记忆能量的点）
        crossover_found = False
        for i in range(1, len(results)):
            if (results[i-1]['W_U_memo_energy'] > results[i-1]['W_U_fourier_energy'] and
                results[i]['W_U_fourier_energy'] > results[i]['W_U_memo_energy']):
                print(f"\n✓ 发现 Crossover 点: Step {results[i]['step']}")
                print(f"  记忆能量={results[i]['W_U_memo_energy']:.4f}, 算法能量={results[i]['W_U_fourier_energy']:.4f}")
                crossover_found = True
                break

        if not crossover_found:
            print("\n✗ 未发现明显的 Crossover 点")

    else:
        print("没有可保存的数据")


def main():
    """主函数 - 支持四种模运算"""
    import argparse

    parser = argparse.ArgumentParser(description='计算电路竞争（记忆 vs 算法）')
    parser.add_argument('--operation', type=str, default='x+y',
                        choices=['x+y', 'x-y', 'x*y', 'x_div_y', 'all'],
                        help='模运算类型')
    parser.add_argument('--n-memo-components', type=int, default=10,
                        help='记忆子空间主成分数')
    parser.add_argument('--n-fourier-freqs', type=int, default=20,
                        help='算法子空间低频分量数')

    args = parser.parse_args()

    # 运算配置
    operations = {
        'x+y': {'name': 'Addition', 'checkpoint_dir': '/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints',
                'output_file': '/root/data1/zjj/Grokking_Formulation/data/x+y/circuit_Competition.csv'},
        'x-y': {'name': 'Subtraction', 'checkpoint_dir': '/root/data1/zjj/Grokking_Formulation/data/x-y/checkpoints',
                'output_file': '/root/data1/zjj/Grokking_Formulation/data/x-y/circuit_Competition.csv'},
        'x*y': {'name': 'Multiplication', 'checkpoint_dir': '/root/data1/zjj/Grokking_Formulation/data/x*y/checkpoints',
                'output_file': '/root/data1/zjj/Grokking_Formulation/data/x*y/circuit_Competition.csv'},
        'x_div_y': {'name': 'Division', 'checkpoint_dir': '/root/data1/zjj/Grokking_Formulation/data/x_div_y/checkpoints',
                    'output_file': '/root/data1/zjj/Grokking_Formulation/data/x_div_y/circuit_Competition.csv'},
    }

    if args.operation == 'all':
        ops_to_process = list(operations.keys())
    else:
        ops_to_process = [args.operation]

    for op in ops_to_process:
        op_config = operations[op]
        checkpoint_dir = op_config['checkpoint_dir']
        output_file = op_config['output_file']

        print("\n" + "=" * 60)
        print(f"处理: {op_config['name']} ({op})")
        print("=" * 60)
        print(f"Checkpoint 目录: {checkpoint_dir}")
        print(f"输出文件: {output_file}")
        print(f"记忆子空间主成分数: {args.n_memo_components}")
        print(f"算法子空间低频分量数: {args.n_fourier_freqs}")
        print("=" * 60)

        if not os.path.exists(checkpoint_dir):
            print(f"跳过: 目录不存在 - {checkpoint_dir}")
            continue

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        compute_circuit_competition(checkpoint_dir, output_file, args.n_memo_components, args.n_fourier_freqs)

    print("\n" + "=" * 60)
    print("全部完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
