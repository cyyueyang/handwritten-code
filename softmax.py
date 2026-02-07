import torch
import torch.nn as nn

def softmax(x: torch.Tensor) -> torch.Tensor:
    max, _ = torch.max(x, dim=-1, keepdim=True)
    x = x - max
    return torch.exp(x) / torch.sum(torch.exp(x), dim=-1, keepdim=True)

if __name__ == '__main__':
    x = torch.randn(10, 10, requires_grad=True)
    y = softmax(x)
    y_true = nn.Softmax(dim=-1)(x)

    if torch.allclose(y, y_true, atol=1e-3):
        print('All tests passed')
