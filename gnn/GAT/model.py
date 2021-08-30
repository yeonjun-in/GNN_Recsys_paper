import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, input_dim, out_dim, device):
        super(GATLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        
        self.W = nn.Linear(input_dim, out_dim)
        self.a = nn.Linear(2*self.out_dim, 1)
        
        self.device = device
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a.weight, gain=1.414)
        
    def forward(self, h, adj):
        # h : (batch_size, input_dim)
        # adj : (batch_size, batch_size) => batch에 해당하는 adj matrix
        batch_size = h.size(0)
        wh = self.W(h) # wh : (batch_size, hidden_dim)

        repeat_wh = wh.repeat_interleave(batch_size, dim=0)
        tile_wh = wh.repeat(batch_size, 1)
        
        wh_concat = torch.cat([repeat_wh, tile_wh], dim=1) # whwh : (batch_size*batch_size, 2*hidden_dim)
        wh_concat = F.leaky_relu(self.a(wh_concat), negative_slope=0.2) # awhwh : (batch_size*batch_size, 1)
        wh_concat = wh_concat.view(batch_size, batch_size, -1).squeeze() # awhwh : (batch_size, batch_size, 1)

        small = -9e15 * torch.ones(batch_size, batch_size).to(self.device)
        
        masked_attention = torch.where(adj > 0, wh_concat, small) # masked_attention : (batch_size, batch_size, n_heads)
        attention_weight = F.softmax(masked_attention, dim=1) # attention_weight : (n_heads, batch_size, batch_size)
        
        return torch.mm(attention_weight, wh).squeeze()

class MultiHeadGATLayer(nn.Module):
    '''
    Attention is all you need 에서 한 방식으로 multihead attention 을 구현해봄
    계속 에러가 나는데 원인을 찾지 못했다... ㅜㅜ
    '''
    def __init__(self, input_dim, out_dim, n_heads, device, concat):
        super(MultiHeadGATLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.concat = concat
        self.W = nn.Linear(input_dim, out_dim)
        self.a = nn.Linear(2*self.n_heads, 1)
        
        self.device = device
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a.weight, gain=1.414)
        
    def forward(self, h, adj):
        # h : (batch_size, input_dim)
        # adj : (batch_size, batch_size) => batch에 해당하는 adj matrix
        batch_size = h.size(0)
        wh = self.W(h) # wh : (batch_size, hidden_dim)
        wh_head = wh.view(batch_size, self.n_heads, self.head_dim)

        repeat_wh = wh_head.repeat_interleave(batch_size, dim=0)
        tile_wh = wh_head.repeat(batch_size, 1, 1)
        
        wh_concat = torch.cat([repeat_wh, tile_wh], dim=2) # whwh : (batch_size*batch_size, 2*hidden_dim)
        wh_concat = F.leaky_relu(self.a(wh_concat), negative_slope=0.2) # awhwh : (batch_size*batch_size, 1)
        wh_concat = wh_concat.view(batch_size, batch_size, -1).squeeze() # awhwh : (batch_size, batch_size, 1)

        small = -9e15 * torch.ones_like(wh_concat).to(self.device)
        adj = adj.repeat(self.n_heads, 1, 1).permute(1,2,0)

        masked_attention = torch.where(adj > 0, wh_concat, small) # masked_attention : (batch_size, batch_size, n_heads)
        attention_weight = F.softmax(masked_attention, dim=1).permute(2,0,1) # attention_weight : (n_heads, batch_size, batch_size)
        
        if self.concat:
            return F.elu(torch.bmm(attention_weight, wh_head.permute(1,0,2)).squeeze()).view(-1, self.out_dim)
        else:
            return torch.bmm(attention_weight, wh_head.permute(1,0,2)).squeeze().mean(dim=0)


class GAT(nn.Module):
    def __init__(self, config):
        super(GAT, self).__init__()
        
        self.multihead_attention = [GATLayer(config.input_dim, config.hidden_dim, config.device) for _ in range(config.n_heads)]
        for i, mha in enumerate(self.multihead_attention):
            self.add_module(f'attention_head{i}', mha)
        
        self.outgat = GATLayer(config.n_heads*config.hidden_dim, config.output_dim, config.device)

        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, h, adj):
        h = self.dropout1(h)
        out = torch.cat([F.elu(mha(h, adj)) for mha in self.multihead_attention], axis=1)
        out = self.dropout1(out)
        out = F.elu(self.outgat(out, adj))
        return out
        