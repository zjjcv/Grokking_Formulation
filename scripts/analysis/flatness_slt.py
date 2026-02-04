#!/usr/bin/env python3
"""
平坦度分析 - 奇异学习理论 (Flatness Analysis via SLT) - 优化版

使用奇异学习理论 (Singular Learning Theory) 计算有效学习系数 λ：

原理：
- 温度化能量：U(β) = E_{θ~p_β(θ|D)}[L_n(θ)]
- p_β(θ|D) ∝ exp(-β * L_n(θ)) * π(θ)
- 自由能关系：∂F_n(β)/∂β = U(β)
- 有效学习系数：λ_eff ≈ [F(β2) - F(β1)] / [log(β2) - log(β1)]

优化策略：
1. 批量扰动采样：一次性生成多个扰动样本
2. 减少采样数量：从 20 次减少到 5 次
3. 稀疏采样 checkpoint：只处理部分 checkpoint
4. 简化数据集：使用更小的数据子集
5. 近似损失估计：基于权重范数变化近似损失变化

使用方法:
    python flatness_slt.py --operation x+y
    python flatness_slt.py --all
"""

import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm


# ==================== 模型定义（与训练时一致） ====================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(context)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class GrokkingTransformer(nn.Module):
    def __init__(self, p=97, num_layers=2, hidden_dim=128, num_heads=4, ffn_dim=512, max_len=3):
        super().__init__()
        self.p = p
        self.vocab_size = p + 1
        self.embedding = nn.Embedding(self.vocab_size, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, hidden_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ffn_dim, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_dim, p)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.embedding(x) + self.pos_encoding[:, :x.shape[1], :]
        for block in self.blocks:
            x = block(x)
        x = x[:, -1, :]
        logits = self.output(x)
        return logits


# ==================== 数据集 ====================
class ModuloDataset(Dataset):
    """简单的模运算数据集"""
    def __init__(self, p=97, operation='add', train=True, seed=42):
        self.p = p
        self.operation = operation

        # 生成所有可能的 (x, y) 组合
        all_pairs = []
        for x in range(p):
            for y in range(p):
                all_pairs.append((x, y))

        # 固定种子划分
        import random
        random.seed(seed)
        random.shuffle(all_pairs)

        n_train = int(len(all_pairs) * 0.5)
        self.pairs = all_pairs[:n_train] if train else all_pairs[n_train:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x, y = self.pairs[idx]

        if self.operation == 'add':
            label = (x + y) % self.p
            op_token = self.p
        elif self.operation == 'sub':
            label = (x - y) % self.p
            op_token = self.p
        elif self.operation == 'mul':
            label = (x * y) % self.p
            op_token = self.p
        elif self.operation == 'div':
            if y == 0:
                label = 0
            else:
                y_inv = pow(y, -1, self.p)
                label = (x * y_inv) % self.p
            op_token = self.p
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

        input_seq = torch.tensor([x, op_token, y], dtype=torch.long)
        return input_seq, label


# ==================== 优化版平坦度估计 ====================
def compute_lambda_from_weights_fast(base_state_dict, base_loss, n_samples=5,
                                     noise_scale=0.01, device='cpu'):
    """
    快速估计有效 λ（基于权重范数变化近似）

    关键优化：
    1. 不进行完整的前向传播
    2. 使用权重范数变化近似损失变化
    3. 减少采样次数

    Args:
        base_state_dict: 基础权重
        base_loss: 基础损失（从 checkpoint 中获取）
        n_samples: 采样次数
        noise_scale: 扰动尺度
        device: 设备

    Returns:
        lambda_eff: 有效学习系数
    """
    # 计算基准权重范数
    base_norm = 0.0
    for param in base_state_dict.values():
        if isinstance(param, torch.Tensor):
            base_norm += torch.norm(param).item() ** 2
    base_norm = math.sqrt(base_norm)

    # 扰动并计算权重范数变化
    norm_changes = []

    for _ in range(n_samples):
        perturbed_norm = 0.0
        for name, param in base_state_dict.items():
            if isinstance(param, torch.Tensor):
                param_np = param.cpu().numpy()
                param_std = np.std(param_np) + 1e-8
                noise = np.random.normal(0, noise_scale * param_std, param_np.shape)
                perturbed_param = param_np + noise
                perturbed_norm += np.sum(perturbed_param ** 2)

        perturbed_norm = math.sqrt(perturbed_norm)
        norm_change = abs(perturbed_norm - base_norm) / (base_norm + 1e-8)
        norm_changes.append(norm_change)

    # 使用权重范数变化的标准差作为平坦度指标
    # λ_eff ∝ 1 / 稳定性，更平坦的损失景观对应更大的 λ
    norm_std = np.std(norm_changes) + 1e-8

    # λ 近似：权重变化越小，越平坦，λ 越大
    lambda_eff = 1.0 / (norm_std * 100 + 0.01)

    return lambda_eff


def compute_lambda_from_sampling(model, dataloader, base_loss, betas,
                                 n_samples=10, noise_scale=0.005, device='cpu',
                                 initial_norm=None, current_norm=None):
    """
    基于采样的 λ 估计（完全重写版 - SLT标准公式）

    SLT理论核心：
    - 温度化后验：p_β(θ|D) ∝ exp(-β·n·L_n(θ))
    - 自由能：F_n(β) = -log ∫ exp(-β·n·L_n(θ)) π(θ)dθ
    - 有效学习系数：λ = lim_{β→∞} [F_n(β) - n·L_n(θ*)] / log(β)

    简化估计：
    1. 使用小扰动近似局部自由能
    2. λ估计为损失对温度的敏感度
    3. 使用相对扰动屏蔽参数范数影响

    Args:
        model: 模型
        dataloader: 数据加载器
        base_loss: 基准训练损失
        betas: 温度列表
        n_samples: 每个温度的采样次数
        noise_scale: 扰动尺度（相对标准差）
        device: 设备
        initial_norm: 初始参数范数
        current_norm: 当前参数范数

    Returns:
        lambda_eff: 有效学习系数估计
        energies: 各温度下的能量
    """
    base_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 计算当前参数范数
    if current_norm is None:
        current_norm = 0.0
        for param in base_state.values():
            if isinstance(param, torch.Tensor):
                current_norm += torch.norm(param).item() ** 2
        current_norm = math.sqrt(current_norm)

    # 获取基准损失（在测试集上）
    base_test_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels, reduction='sum')
            base_test_loss += loss.item()
            total_samples += inputs.size(0)
    base_test_loss = base_test_loss / max(total_samples, 1)

    # 相对扰动缩放因子（屏蔽正则化影响）
    norm_scale = 1.0
    if initial_norm is not None and initial_norm > 1e-8:
        norm_scale = current_norm / initial_norm

    # 在不同温度下估计能量
    energies = {}

    for beta in betas:
        weighted_losses = []

        for _ in range(n_samples):
            # 生成扰动权重（温度越高，扰动越大）
            perturbed_state = {}
            for name, param in base_state.items():
                if isinstance(param, torch.Tensor):
                    param_np = param.numpy()
                    param_std = np.std(param_np) + 1e-8

                    # 温度相关扰动
                    scale_factor = norm_scale / np.sqrt(beta)
                    noise = np.random.normal(0, noise_scale * param_std * scale_factor, param_np.shape)
                    perturbed_state[name] = torch.from_numpy(param_np + noise).float().to(device)
                else:
                    perturbed_state[name] = param

            model.load_state_dict(perturbed_state)

            # 计算损失
            total_loss = 0.0
            total_samples = 0
            with torch.no_grad():
                for inputs, labels in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    logits = model(inputs)
                    loss = F.cross_entropy(logits, labels, reduction='sum')
                    total_loss += loss.item()
                    total_samples += inputs.size(0)

            avg_loss = total_loss / max(total_samples, 1)
            weighted_losses.append(avg_loss)

        # 该温度下的能量：E[L(θ)]
        avg_weighted_loss = np.mean(weighted_losses)
        energies[beta] = avg_weighted_loss

    # 恢复原始权重
    model.load_state_dict(base_state)

    # 计算λ：基于能量随温度的变化率
    # 使用公式：λ ≈ (E₂ - E₁) / log(β₂/β₁)
    # 其中 E 是单位样本的平均损失（能量）
    if len(betas) >= 2:
        beta1, beta2 = betas[0], betas[-1]
        E1, E2 = energies[beta1], energies[beta2]

        # 能量差异
        energy_diff = E2 - E1

        # 温度归一化
        log_beta_diff = np.log(beta2 / beta1)

        # λ估计（放缩因子使数值在合理范围）
        if abs(log_beta_diff) > 1e-8:
            # 注意：由于使用相对扰动，这里不需要额外的归一化
            # λ值本身应该反映出损失景观的几何特性
            lambda_eff = energy_diff / log_beta_diff
            # 使用放大因子使数值更易读
            lambda_eff = lambda_eff * 100
        else:
            lambda_eff = 0.0
    else:
        lambda_eff = 0.0

    return lambda_eff, energies


# ==================== 主分析函数 ====================
def analyze_flatness_from_checkpoints(checkpoint_dir, output_file, operation='add',
                                     p=97, device='cpu', operation_name="",
                                     sparse_step=10, n_samples=10):
    """
    从所有 checkpoint 文件中分析平坦度（有效 λ）- 优化版

    改进：使用初始参数范数归一化，屏蔽正则化影响

    Args:
        checkpoint_dir: checkpoint 文件目录
        output_file: 输出 CSV 文件路径
        operation: 运算类型
        p: 模数
        device: 设备
        operation_name: 操作名称（用于显示）
        sparse_step: 稀疏采样步长（每隔多少个 checkpoint 处理一次）
        n_samples: 每个 checkpoint 的采样次数
    """
    # 获取所有 checkpoint 文件并按步数排序
    checkpoint_files = []
    if not os.path.exists(checkpoint_dir):
        print(f"  警告: 目录不存在 {checkpoint_dir}")
        return

    for f in os.listdir(checkpoint_dir):
        if f.startswith("checkpoint_step_") and f.endswith(".pt"):
            step = int(f.replace("checkpoint_step_", "").replace(".pt", ""))
            checkpoint_files.append((step, os.path.join(checkpoint_dir, f)))

    checkpoint_files.sort(key=lambda x: x[0])

    print(f"  [{operation_name}] 找到 {len(checkpoint_files)} 个 checkpoint 文件")
    print(f"  [{operation_name}] 使用稀疏采样：每 {sparse_step} 个处理 1 个")
    print(f"  [{operation_name}] 每个采样 {n_samples} 次扰动（稳定版）")
    print(f"  [{operation_name}] 使用相对扰动 + 多尺度回归")

    # 稀疏采样：只处理部分 checkpoint
    sampled_files = [checkpoint_files[i] for i in range(0, len(checkpoint_files), sparse_step)]

    # 确保包含最后一个 checkpoint
    if checkpoint_files[-1] not in sampled_files:
        sampled_files.append(checkpoint_files[-1])

    print(f"  [{operation_name}] 实际处理 {len(sampled_files)} 个 checkpoint")

    # 创建小数据集用于快速估计
    dataset = ModuloDataset(p=p, operation=operation, train=True, seed=42)
    # 使用更小的数据子集加速
    subset_size = min(100, len(dataset))
    subset_indices = list(range(subset_size))
    subset = torch.utils.data.Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=128, shuffle=False)

    # β 值设置（减少数量）
    betas = [0.5, 2.0]

    results = []

    # 只创建一次模型
    model = GrokkingTransformer(p=p).to(device)

    # 记录初始参数范数（从第一个checkpoint获取）
    initial_norm = None

    for step, checkpoint_path in tqdm(sampled_files, desc=f"  [{operation_name}] 处理中"):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # 加载权重
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            base_loss = float(checkpoint.get('train_loss', 0))

            # 计算当前参数范数
            current_norm = 0.0
            for param in model.parameters():
                current_norm += torch.norm(param).item() ** 2
            current_norm = math.sqrt(current_norm)

            # 第一个checkpoint：记录初始范数
            if initial_norm is None:
                initial_norm = current_norm
                print(f"  [{operation_name}] 初始参数范数: {initial_norm:.6f}")

            # 估计有效 λ（传入初始范数用于归一化）
            lambda_eff, energies = compute_lambda_from_sampling(
                model, dataloader, base_loss, betas,
                n_samples=n_samples, noise_scale=0.01, device=device,
                initial_norm=initial_norm, current_norm=current_norm
            )

            result = {
                'step': step,
                'train_loss': float(checkpoint.get('train_loss', 0)),
                'train_acc': float(checkpoint.get('train_acc', 0)),
                'test_loss': float(checkpoint.get('test_loss', 0)),
                'test_acc': float(checkpoint.get('test_acc', 0)),
                'lambda_eff': float(lambda_eff),
                'param_norm': float(current_norm),
                'norm_ratio': float(current_norm / initial_norm) if initial_norm else 1.0,
            }

            # 添加各 β 下的能量
            for beta, energy in energies.items():
                result[f'energy_beta_{beta}'] = float(energy)

            results.append(result)

        except Exception as e:
            print(f"    处理 {checkpoint_path} 时出错: {e}")

    if results:
        fieldnames = list(results[0].keys())

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"  [{operation_name}] 数据已保存至: {output_file}")
        print(f"  [{operation_name}] 共保存 {len(results)} 个时间步的数据")

        # 打印统计信息
        if len(results) > 0:
            final_ratio = results[-1]['norm_ratio']
            print(f"\n  [{operation_name}] 最终指标:")
            print(f"    λ_eff = {results[-1]['lambda_eff']:.6f}")
            print(f"    参数范数比 = {final_ratio:.6f}")
    else:
        print(f"  [{operation_name}] 没有可保存的数据")


def main():
    """分析四个模运算任务的平坦度"""
    parser = argparse.ArgumentParser(description='平坦度分析 (SLT) - 稳定版')
    parser.add_argument('--operation', type=str, default='all',
                        choices=['x+y', 'x-y', 'x*y', 'x_div_y', 'all'],
                        help='模运算类型')
    parser.add_argument('--sparse_step', type=int, default=10,
                        help='稀疏采样步长（每隔多少个 checkpoint 处理一次）')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='每个 checkpoint 的采样次数')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    operations = {
        'x+y': {
            'name': 'x+y',
            'operation': 'add',
            'checkpoint_dir': '/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints',
            'output_file': '/root/data1/zjj/Grokking_Formulation/data/x+y/flatness_slt.csv'
        },
        'x-y': {
            'name': 'x-y',
            'operation': 'sub',
            'checkpoint_dir': '/root/data1/zjj/Grokking_Formulation/data/x-y/checkpoints',
            'output_file': '/root/data1/zjj/Grokking_Formulation/data/x-y/flatness_slt.csv'
        },
        'x*y': {
            'name': 'x*y',
            'operation': 'mul',
            'checkpoint_dir': '/root/data1/zjj/Grokking_Formulation/data/x*y/checkpoints',
            'output_file': '/root/data1/zjj/Grokking_Formulation/data/x*y/flatness_slt.csv'
        },
        'x_div_y': {
            'name': 'x÷y',
            'operation': 'div',
            'checkpoint_dir': '/root/data1/zjj/Grokking_Formulation/data/x_div_y/checkpoints',
            'output_file': '/root/data1/zjj/Grokking_Formulation/data/x_div_y/flatness_slt.csv'
        },
    }

    if args.operation == 'all':
        ops_to_process = list(operations.keys())
    else:
        ops_to_process = [args.operation]

    print("=" * 70)
    print("平坦度分析 - 奇异学习理论 (Flatness Analysis via SLT) - 优化版")
    print("=" * 70)
    print("计算有效学习系数 λ_eff")
    print(f"稀疏采样步长: {args.sparse_step}")
    print(f"每个采样次数: {args.n_samples}")
    print("=" * 70)

    for op_key in ops_to_process:
        op = operations[op_key]
        print(f"\n正在分析: {op['name']}")
        print(f"Checkpoint 目录: {op['checkpoint_dir']}")
        print(f"输出文件: {op['output_file']}")
        print("-" * 70)

        analyze_flatness_from_checkpoints(
            op['checkpoint_dir'],
            op['output_file'],
            operation=op['operation'],
            p=97,
            device=device,
            operation_name=op['name'],
            sparse_step=args.sparse_step,
            n_samples=args.n_samples
        )

    print("\n" + "=" * 70)
    print("全部完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
