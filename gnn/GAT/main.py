from utils import load_data, seed_everything
from model import GATLayer, MultiHeadGATLayer, GAT

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from collections import defaultdict
from datetime import datetime


train_mask, val_mask, test_mask, feat, label, adj = load_data()

batch_size = len(train_mask)
input_dim = feat.shape[1]
output_dim = label.unique().shape[0]

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1995, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--input_dim', type=int, default=input_dim, help='Number of input units.')
parser.add_argument('--hidden_dim', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--output_dim', type=int, default=output_dim, help='Number of output units.')
parser.add_argument('--n_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--bs', type=int, default=batch_size, help='Batch Size')
parser.add_argument('--device', type=str, default='cuda:3', help='Device GPU')

args = parser.parse_args()

seed_everything(args.seed)

model = GAT(args)
model = model.to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss_fn = nn.CrossEntropyLoss()

if 'cuda' in args.device:
    feat = feat.to(args.device)
    label = label.to(args.device)
    adj = adj.to(args.device)
    train_mask = train_mask.to(args.device)
    test_mask = test_mask.to(args.device)
    
train_label = label[train_mask]
test_label = label[test_mask]


history = defaultdict(list)
start = datetime.now()
best_loss, early_step, best_epoch = 0, 0, 0
for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()
    output = model(feat, adj)
    acc = torch.sum(train_label == torch.argmax(output[train_mask], axis=1)) / len(train_label)
    loss = loss_fn(output[train_mask], train_label)
    loss.backward()
    optimizer.step()

    history['train_loss'].append(loss.item())
    history['train_acc'].append(acc)

    model.eval()
    with torch.no_grad():    
        output = model(feat, adj)
        acc = torch.sum(test_label == torch.argmax(output[test_mask], axis=1)) / len(test_label)
        loss = loss_fn(output[test_mask], test_label)

    history['valid_loss'].append(loss.item())
    history['valid_acc'].append(acc)

    if epoch == 0 or epoch == args.epochs-1 or (epoch+1)%10 == 0:
        print(f'EPOCH {epoch+1} : TRAINING loss {history["train_loss"][-1]:.3f}, TRAINING ACC {history["train_acc"][-1]:.3f}, VALID loss {history["valid_loss"][-1]:.3f}, VALID ACC {history["valid_acc"][-1]:.3f}')
    
    if history['valid_acc'][-1] > best_loss:
        best_loss = history['valid_acc'][-1]
        best_epoch = epoch

end = datetime.now()
print(end-start)
print(f'At EPOCH {best_epoch + 1}, We have Best Acc {best_loss}')
