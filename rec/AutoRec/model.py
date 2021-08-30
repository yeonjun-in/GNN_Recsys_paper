import torch.nn as nn
import torch.nn.functional as F


class AutoRec(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AutoRec, self).__init__()
        self.enc = nn.Linear(input_dim, hidden_dim)
        self.dec = nn.Linear(hidden_dim, output_dim)
        self.activate = F.sigmoid


    def forward(self, x):
        x = self.activate(self.enc(x))
        x = self.dec(x)
        return x
