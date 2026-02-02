#!/usr/bin/env python3
"""
相位干预实验：验证"幅度已就绪，只缺相位"假设

对于每个 checkpoint：
1. 提取输入嵌入 W_e (p, d_model)
2. 对 W_e 进行列向 DFT 得到 F(W_e) = A * exp(i*Phi)
3. 构造"理想相位"：phi_hat[k, x] = 2*pi*k*x/p (线性相位分布)
4. 保留原始幅度，结合理想相位重构
5. 逆 DFT 得到 W_e_intervened
6. 替换回模型评估准确率

预期：在 Grokking 前的平台期，干预后准确率应显著高于原始准确率
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm


# ==================== 模型架构（与训练时完全一致）====================
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

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

        # Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)

        # Output
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(context)


class FeedForward(nn.Module):
    """前馈神经网络"""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class TransformerBlock(nn.Module):
    """Transformer 编码器块"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class GrokkingTransformer(nn.Module):
    """用于模运算的 Transformer 模型（与训练时完全一致）"""

    def __init__(self, p, d_model=128, num_heads=4, num_layers=2, d_ff=512, dropout=0.1):
        super().__init__()
        self.p = p
        self.d_model = d_model

        # Token embedding (0 到 p-1 是数字，p 是操作符)
        self.vocab_size = p + 1
        self.embedding = nn.Embedding(self.vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 3, d_model))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Output head (映射到 p 个类)
        self.output = nn.Linear(d_model, p)

    def forward(self, x):
        # x: (batch, seq_len)
        # Embedding
        x = self.embedding(x) + self.pos_encoding[:, :x.shape[1], :]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # 取最后一个 token 的输出
        x = x[:, -1, :]

        # Output
        logits = self.output(x)

        return logits


class ModuloDataset(torch.utils.data.Dataset):
    def __init__(self, p, split='train', train_ratio=0.5, seed=42):
        self.p = p
        self.data = []

        # 生成所有可能的 (x, y) 组合
        all_pairs = []
        for x in range(p):
            for y in range(p):
                all_pairs.append((x, y))

        # 使用固定随机种子进行划分
        import random
        random.seed(seed)
        random.shuffle(all_pairs)

        n_train = int(len(all_pairs) * train_ratio)
        if split == 'train':
            self.pairs = all_pairs[:n_train]
        else:
            self.pairs = all_pairs[n_train:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x, y = self.pairs[idx]
        op = self.p  # 操作符 token
        return torch.tensor([x, op, y]), (x + y) % self.p


def create_ideal_phase_matrix(p, d_model):
    """
    构造理想相位矩阵

    对于 DFT 后的矩阵 F(W_e) (p, d_model)，
    其中行对应频率 k，列对应特征 j
    理想相位：phi_hat[k, j] = 2*pi*k*j/p (线性相位)

    Args:
        p: 模数
        d_model: 嵌入维度

    Returns:
        phi_hat: (p, d_model) 理想相位矩阵
    """
    k = np.arange(p).reshape(-1, 1)  # 频率索引 (p, 1)
    j = np.arange(d_model)  # 特征索引 (d_model)

    phi_hat = 2 * np.pi * k * j / p  # (p, d_model)

    return phi_hat


def apply_phase_intervention(W_e, ideal_phase):
    """
    对 W_e 应用相位干预

    1. 对每列进行 DFT
    2. 保留幅度，替换为理想相位
    3. 逆 DFT 重构

    Args:
        W_e: (p, d_model) 嵌入矩阵
        ideal_phase: (p, d_model) 理想相位矩阵

    Returns:
        W_e_intervened: (p, d_model) 干预后的嵌入矩阵
    """
    p, d_model = W_e.shape

    # 对每列进行 DFT
    W_e_dft = np.fft.fft(W_e, axis=0)  # (p, d_model)

    # 提取幅度
    magnitudes = np.abs(W_e_dft)  # (p, d_model)

    # 使用理想相位重构
    W_e_dft_intervened = magnitudes * np.exp(1j * ideal_phase)  # (p, d_model)

    # 逆 DFT 得到实数矩阵
    W_e_intervened = np.real(np.fft.ifft(W_e_dft_intervened, axis=0))  # (p, d_model)

    return W_e_intervened


def evaluate_accuracy(model, dataloader, device):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    return correct / total if total > 0 else 0.0


def extract_phase_intervention_from_checkpoints(checkpoint_dir, output_file):
    """从所有 checkpoint 文件中提取相位干预实验数据"""
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("checkpoint_step_") and f.endswith(".pt"):
            step = int(f.replace("checkpoint_step_", "").replace(".pt", ""))
            checkpoint_files.append((step, os.path.join(checkpoint_dir, f)))

    checkpoint_files.sort(key=lambda x: x[0])

    print(f"找到 {len(checkpoint_files)} 个 checkpoint 文件")

    # 创建理想相位矩阵
    p = 97
    d_model = 128
    ideal_phase = create_ideal_phase_matrix(p, d_model)
    print(f"理想相位矩阵形状: {ideal_phase.shape}")

    # 创建测试数据集
    test_dataset = ModuloDataset(p, split='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False
    )
    print(f"测试集大小: {len(test_dataset)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    results = []

    for step, checkpoint_path in tqdm(checkpoint_files, desc="处理 checkpoints"):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # 创建模型并加载权重
            model = GrokkingTransformer(p=p).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])

            # 评估原始模型
            original_acc = evaluate_accuracy(model, test_loader, device)

            # 提取 W_e 并应用相位干预
            W_e = checkpoint['model_state_dict']['embedding.weight'].cpu().numpy()
            W_e_p = W_e[:p, :]  # 只取前 p 行

            W_e_intervened = apply_phase_intervention(W_e_p, ideal_phase)

            # 替换模型中的嵌入
            with torch.no_grad():
                model.embedding.weight[:p, :] = torch.tensor(
                    W_e_intervened, dtype=torch.float32, device=device
                )

            # 评估干预后的模型
            intervened_acc = evaluate_accuracy(model, test_loader, device)

            result = {
                'step': step,
                'train_loss': checkpoint['train_loss'],
                'train_acc': checkpoint['train_acc'],
                'test_loss': checkpoint['test_loss'],
                'test_acc': checkpoint['test_acc'],
                'original_test_acc': float(original_acc),
                'intervened_test_acc': float(intervened_acc),
                'acc_improvement': float(intervened_acc - original_acc),
            }

            results.append(result)

            if step % 100 == 0:
                print(f"Step {step}: Original={original_acc:.4f}, "
                      f"Intervened={intervened_acc:.4f}, "
                      f"Improvement={intervened_acc - original_acc:.4f}")

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

        # 打印统计信息
        improvements = [r['acc_improvement'] for r in results]
        print(f"\n准确率改进统计:")
        print(f"  平均改进: {np.mean(improvements):.4f}")
        print(f"  最大改进: {np.max(improvements):.4f}")
        print(f"  最小改进: {np.min(improvements):.4f}")
    else:
        print("没有可保存的数据")


def main():
    checkpoint_dir = "/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints"
    output_file = "/root/data1/zjj/Grokking_Formulation/data/x+y/phase_intervention.csv"

    print("=" * 60)
    print("相位干预实验：验证'幅度已就绪，只缺相位'假设")
    print("=" * 60)
    print(f"Checkpoint 目录: {checkpoint_dir}")
    print(f"输出文件: {output_file}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    extract_phase_intervention_from_checkpoints(checkpoint_dir, output_file)


if __name__ == "__main__":
    main()
