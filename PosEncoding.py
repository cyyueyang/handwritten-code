import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    E[pos, 2i] = sin(pos/(10000^(2i/d_model)))
    E[pos, 2i + 1] = cos(pos/(10000^(2i/d_model)))
    """
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        bs, seq_len, d_model = x.size()
        x = x + self.pe[:, :seq_len, :]
        return x

    def get_encoding(self, seq_len):
        return self.pe[:, :seq_len, :]

class RotaryPositionalEncoding(nn.Module):
    """
    支持
    [batch_size, seq_len, d_model]
    [batch_size, n_heads, seq_len, d_model]
    """
    def __init__(self, d_model, max_len=5000, base=10000.0):
        super(RotaryPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base

        inv_freq = 1. / (base ** (torch.arange(0, d_model, 2).float() / self.d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        seq_len = x.size(-2)
        t = torch.arange(0, seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)

        if x.dim() == 4:
            freqs = freqs.unsqueeze(0).unsqueeze(0)
        else:
            freqs = freqs.unsqueeze(0)

        cos_cached = freqs.cos()
        sin_cached = freqs.sin()

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        rotate1 = x1 * cos_cached - x2 * sin_cached
        rotate2 = x2 * cos_cached + x1 * sin_cached

        rotated = torch.stack((rotate1, rotate2), dim=-1).flatten(-2)

        return rotated

    def get_encoding_matrix(self, seq_len):
        t = torch.arange(0, seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, d_model/2]

        # 扩展为完整维度 [seq_len, d_model]
        emb_cos = torch.cos(freqs).repeat_interleave(2, dim=1)
        emb_sin = torch.sin(freqs).repeat_interleave(2, dim=1)

        # 构建输出：偶数维放cos，奇数维放sin
        encoding = torch.zeros(seq_len, self.d_model)
        encoding[:, ::2] = emb_cos[:, ::2]  # 偶数维
        encoding[:, 1::2] = emb_sin[:, 1::2]  # 奇数维
        return encoding


def visualize_positional_encodings():
    d_model = 64
    seq_len = 50

    # 使用相同维度初始化两种编码
    sin_pe = SinusoidalPositionalEncoding(d_model, max_len=seq_len)
    rope_pe = RotaryPositionalEncoding(d_model, max_len=seq_len)  # 传入64而非8

    # 获取编码矩阵
    sinusoidal_matrix = sin_pe.get_encoding(seq_len).squeeze(0).numpy()
    rope_matrix = rope_pe.get_encoding_matrix(seq_len).numpy()  # 不传d_model参数

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 正弦编码热力图
    ax1 = axes[0, 0]
    im1 = ax1.imshow(sinusoidal_matrix.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_title('Sinusoidal Positional Encoding', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Dimension')
    plt.colorbar(im1, ax=ax1)

    # 2. RoPE热力图
    ax2 = axes[0, 1]
    im2 = ax2.imshow(rope_matrix.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_title('RoPE (Rotary Positional Embedding)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Dimension')
    plt.colorbar(im2, ax=ax2)

    # 3. 特定维度的波形对比
    ax3 = axes[1, 0]
    dims = [0, 1, 10, 11, 30, 31]
    for dim in dims:
        ax3.plot(sinusoidal_matrix[:, dim], label=f'Dim {dim}', linewidth=2)
    ax3.set_title('Sinusoidal: Waveforms by Dimension', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. RoPE波形
    ax4 = axes[1, 1]
    for dim in dims:
        ax4.plot(rope_matrix[:, dim], label=f'Dim {dim}', linewidth=2)
    ax4.set_title('RoPE: Waveforms by Dimension', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('positional_encoding_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_usage():
    batch_size = 2
    seq_len = 10
    d_model = 64
    num_heads = 8
    head_dim = d_model // num_heads

    x = torch.randn(batch_size, seq_len, d_model)

    print("=" * 50)
    print("正弦位置编码使用示例")
    print("=" * 50)
    sin_pe = SinusoidalPositionalEncoding(d_model)
    output = sin_pe(x)
    print(f"输入形状:  {x.shape}")
    print(f"输出形状:  {output.shape}")

    print("\n" + "=" * 50)
    print("RoPE使用示例 (多头注意力场景)")
    print("=" * 50)

    # 模拟Q/K投影 [batch, heads, seq, head_dim]
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # 注意：RoPE对每个head单独应用，所以传入head_dim
    rope = RotaryPositionalEncoding(head_dim)
    Q_rotated = rope(Q)
    K_rotated = rope(K)

    print(f"原始Q形状:    {Q.shape}")
    print(f"旋转后Q形状:  {Q_rotated.shape}")

    # 计算注意力分数
    scores = torch.matmul(Q_rotated, K_rotated.transpose(-2, -1)) / math.sqrt(head_dim)
    print(f"注意力分数:   {scores.shape}")

if __name__ == "__main__":
    visualize_positional_encodings()
    demo_usage()

