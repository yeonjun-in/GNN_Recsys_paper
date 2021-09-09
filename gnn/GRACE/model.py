import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils import cosine_similarity
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(config.input_dim, config.hidden_dim)
        self.conv2 = GCNConv(config.hidden_dim, config.hidden_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.w1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.w2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        torch.nn.init.xavier_uniform_(self.w1.weight.data)
        torch.nn.init.xavier_uniform_(self.w2.weight.data)
    
    def forward(self, x):
        x = F.relu(self.w1(x))
        x = self.w2(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.temper = config.temper
        self.pos_mask = torch.diag(torch.ones(config.nb_nodes)).to(config.device)
        self.neg_mask = 1 - self.pos_mask
    
    def forward(self, u, v):
        inter_view = torch.exp(cosine_similarity(u, v)/self.temper) 
        intra_view = torch.exp(cosine_similarity(u, u)/self.temper)
        
        pos = inter_view * self.pos_mask
        neg_inter = inter_view * self.neg_mask
        neg_intra = intra_view * self.neg_mask

        loss =  torch.log(pos.diagonal(0) / (pos+neg_inter+neg_intra).sum(axis=1))
        return loss


class GRACE(nn.Module):
    def __init__(self, config):
        super(GRACE, self).__init__()
        self.gcn = GCN(config)
        self.mlp = MLP(config)
        self.disc = Discriminator(config)

    def forward(self, x_1, adj_1, x_2, adj_2):
        u = self.gcn(x_1, adj_1)
        v = self.gcn(x_2, adj_2)
        u = self.mlp(u)
        v = self.mlp(v)
        loss1 = self.disc(u, v)
        loss2 = self.disc(v, u)
        return -torch.mean((loss1+loss2)/2)

    def get_embed(self, x, adj):
        embed = self.gcn(x, adj)
        embed = self.mlp(embed)
        return embed.detach().cpu().numpy()
