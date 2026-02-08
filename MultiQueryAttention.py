import torch
import torch.nn.functional as F
import math
import torch.nn as nn

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int=2048):
        super(MultiQueryAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, 2*self.head_dim, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        self.register_buffer('mask', mask)

    def forward(self, x):
        bs, seq_len, d_model = x.size()

        q = self.q(x)
        kv = self.kv(x)
        q = q.view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = kv.chunk(2, dim=-1)
        k = k.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        v = v.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(~self.mask.unsqueeze(0).unsqueeze(0)[:, :, :seq_len, :seq_len], -1e9)
        attn = attn.softmax(dim=-1)
        attn_score = attn @ v
        attn_score = attn_score.transpose(1, 2).contiguous().view(bs, seq_len, d_model)
        out = self.w_o(attn_score)
        return out

if __name__ == '__main__':
    x = torch.randn(1, 128, 256)
    text_mqa = MultiQueryAttention(d_model=256, n_heads=8, max_len=512)
    y = text_mqa(x)
    print(y.shape)
