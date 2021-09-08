import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils import corrupt_fn

class GCNLayer(nn.Module):
    def __init__(self, input, output):
        super(GCNLayer, self).__init__()
        self.input = input
        self.output = output
        self.W = nn.Linear(input, output, bias=False)
        torch.nn.init.xavier_uniform_(self.W.weight.data)
        self.act = nn.PReLU()
    
    def forward(self, x, adj):
        output = torch.spmm(adj, x)
        output = self.W(output)
        return self.act(output)

class GCN(nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        self.conv1 = GCNLayer(config.input_dim, config.hidden_dim)
        
    def forward(self, x, adj):
        x = self.conv1(x, adj)
        return x

class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()
        self.sigmoid = nn.Sigmoid()
    def forward(self, h):
        return self.sigmoid(torch.mean(h, dim=0)).unsqueeze(0)
        

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.w = nn.Bilinear(config.hidden_dim, config.hidden_dim, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.w.weight.data)
    
    def forward(self, h, s):
        return self.w(h, s)

class DGI(nn.Module):
    def __init__(self, config):
        super(DGI, self).__init__()
        self.gcn = GCN(config)
        self.readout = Readout()
        self.disc = Discriminator(config)
    
    def forward(self, x, x_tilde, adj):
        h = self.gcn(x, adj)
        h_tilde = self.gcn(x_tilde, adj)

        s = self.readout(h).expand_as(h)

        dp = self.disc(h, s).squeeze().unsqueeze(0)
        dn = self.disc(h_tilde, s).squeeze().unsqueeze(0)
        
        return torch.cat((dp, dn), axis=1)

    def get_embed(self, x, adj):
        h = self.gcn(x, adj)
        s = self.readout(h)
        return h.detach().cpu().numpy(), s.detach().cpu().numpy()
