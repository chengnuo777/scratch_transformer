from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    """
    标准的 线性层 + softmax层
    """
    def __init__(self, d_model, d_vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, d_vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)