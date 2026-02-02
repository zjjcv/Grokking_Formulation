#!/usr/bin/env python3
"""
频谱归因分析：计算不同频率分量对 Logit 预测的贡献

核心思想：
1. 将 W_e 分解为不同频率分量：W_e = Σ_k W_e^(k)
2. 对于每个频率 k，计算其对正确标签的 Logit 贡献
3. 统计低频 vs 高频贡献的演化

预期：Grokking 时，特定频率的贡献突然爆发
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# ==================== 模型架构（与训练时完全一致）====================
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
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
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
    def __init__(self, p, d_model=128, num_heads=4, num_layers=2, d_ff=512, dropout=0.1):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.vocab_size = p + 1
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 3, d_model))
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(d_model, p)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.shape[1], :]
        for block in self.blocks:
            x = block(x)
        x = x[:, -1, :]
        return self.output(x)


def decompose_embedding_by_frequency(W_e, p):
    """
    将嵌入矩阵分解为不同频率分量

    Args:
        W_e: (vocab_size, d_model) 嵌入矩阵
        p: 模数

    Returns:
        frequency_components: List of (p, d_model) arrays, each corresponding to a frequency
    """
    W_e_p = W_e[:p, :]  # (p, d_model)

    # 对每列进行 DFT
    W_e_dft = np.fft.fft(W_e_p, axis=0)  # (p, d_model) 复数

    # 为每个频率重建时域分量
    components = []
    for k in range(p):
        # 创建只有频率 k 的频谱
        freq_spectrum = np.zeros_like(W_e_dft, dtype=complex)
        freq_spectrum[k, :] = W_e_dft[k, :]

        # 逆 DFT 得到该频率的时域分量
        component = np.fft.ifft(freq_spectrum, axis=0).real  # (p, d_model)

        # 扩展回完整 vocab_size（操作符 token 设为 0）
        full_component = np.zeros_like(W_e, dtype=float)
        full_component[:p, :] = component

        components.append(full_component)

    return components


def compute_frequency_attribution(model, W_e_original, frequency_components, test_loader, device, p, max_freq=20):
    """
    计算每个频率分量对正确 Logit 的贡献

    简化策略：直接测量替换嵌入后的 Logit 变化

    Args:
        model: 模型
        W_e_original: 原始嵌入权重
        frequency_components: 各频率分量列表
        test_loader: 测试数据加载器
        device: 计算设备
        p: 模数
        max_freq: 分析的最大频率

    Returns:
        attributions: 每个频率的平均贡献值
    """
    model.eval()

    # 保存原始嵌入
    original_W_e = model.embedding.weight.data.clone()

    # 计算原始模型的正确 Logit（基准）
    original_correct_logits = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            correct_logits = outputs[range(len(targets)), targets.cpu()]
            original_correct_logits.append(correct_logits.cpu())
    original_correct_logits = torch.cat(original_correct_logits)

    # 对每个频率计算贡献
    attributions = {}

    # 只分析前 max_freq 个频率（低频最重要）
    for k in range(min(max_freq, len(frequency_components))):
        freq_component = frequency_components[k]

        # 替换为只有该频率分量的嵌入
        with torch.no_grad():
            model.embedding.weight.data = torch.tensor(
                freq_component, dtype=torch.float32, device=device
            )

        # 计算该频率下的正确 Logit
        freq_correct_logits = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                correct_logits = outputs[range(len(targets)), targets.cpu()]
                freq_correct_logits.append(correct_logits.cpu())
        freq_correct_logits = torch.cat(freq_correct_logits)

        # 计算贡献：该频率分量的 Logit 与原始 Logit 的相关性
        # 或者直接使用平均 Logit 值
        attribution = float(freq_correct_logits.mean().item())
        attributions[k] = attribution

    # 恢复原始嵌入
    model.embedding.weight.data = original_W_e

    return attributions


def compute_spectral_attribution_simple(W_e, p, max_freq=20):
    """
    简化版频谱归因：直接分析频率域的能量分布

    不需要运行模型，直接计算每个频率的能量

    Args:
        W_e: (vocab_size, d_model) 嵌入矩阵
        p: 模数
        max_freq: 分析的最大频率

    Returns:
        attributions: 每个频率的能量值
    """
    W_e_p = W_e[:p, :]  # (p, d_model)

    # 对每列进行 DFT
    W_e_dft = np.fft.fft(W_e_p, axis=0)  # (p, d_model) 复数

    # 计算每个频率的能量（幅度平方）
    power_spectrum = np.abs(W_e_dft) ** 2  # (p, d_model)

    # 对特征维度求平均，得到每个频率的总能量
    freq_energy = power_spectrum.mean(axis=1)  # (p,)

    attributions = {}
    for k in range(min(max_freq, p)):
        attributions[k] = float(freq_energy[k])

    return attributions


def extract_spectral_attribution_from_checkpoints(checkpoint_dir, output_file):
    """从所有 checkpoint 文件中提取频谱归因数据"""
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("checkpoint_step_") and f.endswith(".pt"):
            step = int(f.replace("checkpoint_step_", "").replace(".pt", ""))
            checkpoint_files.append((step, os.path.join(checkpoint_dir, f)))

    checkpoint_files.sort(key=lambda x: x[0])

    print(f"找到 {len(checkpoint_files)} 个 checkpoint 文件")

    p = 97
    max_freq = 20  # 分析前 20 个频率（低频）

    results = []

    for step, checkpoint_path in tqdm(checkpoint_files, desc="处理 checkpoints"):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            W_e = checkpoint['model_state_dict']['embedding.weight'].numpy()

            # 使用简化版频谱归因
            attributions = compute_spectral_attribution_simple(W_e, p, max_freq)

            result = {
                'step': step,
                'train_loss': checkpoint['train_loss'],
                'train_acc': checkpoint['train_acc'],
                'test_loss': checkpoint['test_loss'],
                'test_acc': checkpoint['test_acc'],
            }

            # 添加各频率的归因值
            for k, attr in attributions.items():
                result[f'freq_{k}_attribution'] = attr

            # 计算低频和高频贡献
            low_freq_sum = sum(attributions[k] for k in range(min(6, max_freq)))  # k=0 到 5
            high_freq_sum = sum(attributions[k] for k in range(6, max_freq))

            result['low_freq_attribution'] = low_freq_sum
            result['high_freq_attribution'] = high_freq_sum
            result['low_high_ratio'] = low_freq_sum / (high_freq_sum + 1e-10)

            results.append(result)

            if step % 100 == 0:
                print(f"Step {step}: Low={low_freq_sum:.4f}, High={high_freq_sum:.4f}, "
                      f"Ratio={result['low_high_ratio']:.4f}")

        except Exception as e:
            print(f"处理 {checkpoint_path} 时出错: {e}")
            import traceback
            traceback.print_exc()

    if results:
        fieldnames = list(results[0].keys())
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n数据已保存至: {output_file}")
        print(f"共保存 {len(results)} 个时间步的数据")
    else:
        print("没有可保存的数据")


def main():
    checkpoint_dir = "/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints"
    output_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/spectral_attribution.csv"

    print("=" * 60)
    print("频谱归因分析：不同频率分量对 Logit 的贡献")
    print("=" * 60)
    print(f"Checkpoint 目录: {checkpoint_dir}")
    print(f"输出文件: {output_file}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    extract_spectral_attribution_from_checkpoints(checkpoint_dir, output_file)


if __name__ == "__main__":
    main()
