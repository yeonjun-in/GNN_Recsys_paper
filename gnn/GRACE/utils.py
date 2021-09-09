import numpy as np
import torch
import torch.nn.functional as F
import random, os
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dropout_adj

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_data():
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    graph = dataset[0]
    feat, label, train_mask, test_mask = graph.x, graph.y, graph.train_mask, graph.test_mask
    return graph, graph.edge_index, feat, label.numpy(), train_mask.numpy(), test_mask.numpy()

def corrupt_fn(x, adj, p_r, p_m):
    adj = dropout_adj(adj, p=p_r)[0]
    node_feat_mask = torch.rand_like(x[[0], :]).expand_as(x).to(x.device)
    node_feat_mask = torch.where(node_feat_mask > p_m, 1.0, 0.0)
    return x * node_feat_mask, adj


def cosine_similarity(a, b):
    a, b = F.normalize(a), F.normalize(b) 
    return F.linear(a, b)

class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    