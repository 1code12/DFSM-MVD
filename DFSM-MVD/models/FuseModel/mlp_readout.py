import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation
"""


import torch.nn as nn

class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim // 2, bias=True)
        self.linear2 = nn.Linear(input_dim // 2, output_dim, bias=True)

    def forward(self, x):
        y = self.linear1(x)
        y = F.relu(y)
        y = F.dropout(y, p=0.5)
        y = self.linear2(y)
        return y

