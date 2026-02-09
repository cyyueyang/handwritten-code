import torch
import torch.nn.functional as F
import math
import torch.nn as nn

class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_groups, max_len=2048):
        super(GroupQueryAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_groups = n_groups
        assert d_model % self.n_heads == 0
        assert n_heads % n_groups == 0
        self.head_dim = d_model // self.n_heads

        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2*self.n_groups*self.head_dim)
        self.w_o = nn.Linear(d_model, d_model)

        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        self.register_buffer('mask', mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        bs, seq_len, d_model = x.size()
        q = self.q(x)
        k, v = self.kv(x).chunk(2, dim=-1)
        q = q.view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, seq_len, self.n_groups, self.head_dim).transpose(1, 2)
        v = v.view(bs, seq_len, self.n_groups, self.head_dim).transpose(1, 2)

        k = self.repeat(k, self.n_heads // self.n_groups)
        v = self.repeat(v, self.n_heads // self.n_groups)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(~self.mask[:, :, :seq_len, :seq_len], -1e9)
        attn = attn.softmax(dim=-1)
        attn_score = attn @ v
        attn_score = attn_score.transpose(1, 2).contiguous().view(bs, seq_len, d_model)
        out = self.w_o(attn_score)
        return out

    @staticmethod
    def repeat(x, rep):
        bs, n_groups, seq_len, head_model = x.size()
        x = x.unsqueeze(2)
        x = x.expand(-1, -1, rep, -1, -1).contiguous()
        x = x.view(bs, n_groups*rep, seq_len, head_model)
        return x

if __name__ == '__main__':
    x = torch.randn(8, 32, 128)
    text_gqa = GroupQueryAttention(128, 16, 8, 2048)
    y = text_gqa(x)
    print(y.size())



