import torch.nn as nn
from .layer_norm import LayerNorm
import pdb


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # pdb.set_trace()
        # 有问题
        "Apply residual connection to any sublayer with the same size."
        # result = x + self.dropout(sublayer(self.norm(x)))
        # return x + sublayer(self.norm(x))

        return x + self.dropout(sublayer(self.norm(x)))
