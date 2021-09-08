import numpy as np
import torch 
import torch.nn as nn
from torch.random import seed
from utils import load_data, corrupt_fn, AverageMeter, get_A_mat, seed_everything
from model import DGI
from sklearn.linear_model import LogisticRegression

class Config:
    lr = 0.001
    weight_decay = 0
    hidden_dim = 512
    epochs = 100
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

config = Config()

graph, adj, x, y, train_mask, test_mask = load_data()
config.input_dim = x.shape[1]
label = torch.cat((torch.ones(x.size(0)), torch.zeros(x.size(0))))

adj = get_A_mat(graph, config)
x = x.to(config.device)
label = label.to(config.device)

loss_fn = nn.BCEWithLogitsLoss()
model = DGI(config)
model = model.to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

seed_everything(1995)

losses = AverageMeter()
best_loss, best_epoch = np.inf, 0
for epoch in range(config.epochs):
    x_tilde = corrupt_fn(x).to(config.device)

    model.train()
    optimizer.zero_grad()
    out = model(x, x_tilde, adj)
    loss = loss_fn(out, label.unsqueeze(0))
    losses.update(loss.item())
    
    if best_loss > loss.item():
        best_loss = loss.item()
        best_epoch = epoch+1
        torch.save(model.state_dict(), f'/home/yeonjun/Desktop/gnn/DGI/self_sup_weight.pth')

    loss.backward()
    optimizer.step()
    print(f'EPOCH {epoch+1}: Train Loss {loss.item():.5f}')

print(f'Best Loss {best_loss:.5f} at Epoch {best_epoch}')


node_embed, graph_embed = model.get_embed(x, adj)

X_train, X_test, y_train, y_test = node_embed[train_mask], node_embed[test_mask], y[train_mask], y[test_mask]
clf = LogisticRegression(random_state=1995, solver='liblinear')
clf.fit(X_train, y_train)
print('Downstream Task Accuracy:', np.sum(clf.predict(X_test) == y_test)/len(y_test))

