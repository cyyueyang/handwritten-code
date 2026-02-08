import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len=2048):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        self.register_buffer('mask', mask)
    def forward(self, x):
        bs, seq_len, d_model = x.size()
        mask = self.mask.unsqueeze(0).unsqueeze(0)[:, :, :seq_len, :seq_len]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(~mask, -1e9)
        attn = attn.softmax(dim=-1)
        attn_score = attn @ v
        attn_score = attn_score.transpose(1, 2).contiguous().view(bs, seq_len, d_model)
        out = self.w_o(attn_score)
        return out

if __name__ == "__main__":
    x = torch.randn(32, 128, 256)
    test_mha = MultiHeadAttention(256, 4)
    out = test_mha(x)
    print(out.size())