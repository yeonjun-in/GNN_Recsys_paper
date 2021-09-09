import torch
from torch.optim import Adam
from utils import load_data, corrupt_fn, AverageMeter, seed_everything
from model import GRACE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

class Config:
    lr = 0.001
    weight_decay = 5e-4
    epochs = 200
    hidden_dim = 128
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    temper = 0.4
    
config = Config()

seed_everything(1995)

graph, adj, x, y, train_mask, test_mask = load_data()
config.input_dim = x.size(1)
config.nb_nodes = x.size(0)
model = GRACE(config)

if torch.cuda.is_available():
    x = x.to(config.device)
    adj = adj.to(config.device)
    model = model.to(config.device)

optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

for epoch in range(config.epochs):
    x_1, adj_1 = corrupt_fn(x, adj, 0.2, 0.3)
    x_2, adj_2 = corrupt_fn(x, adj, 0.4, 0.4)
    
    model.train()
    optimizer.zero_grad()

    loss = model(x_1, adj_1, x_2, adj_2)
    loss.backward()
    optimizer.step()
    print(F'EPOCH {epoch+1}: Loss {loss.item()}')

node_embed = model.get_embed(x, adj)

X_train, X_test, y_train, y_test = node_embed[train_mask], node_embed[test_mask], y[train_mask], y[test_mask]
clf = LogisticRegression(random_state=1995, solver='liblinear')
clf.fit(X_train, y_train)
micro = f1_score(y_test, clf.predict(X_test), average="micro")
macro = f1_score(y_test, clf.predict(X_test), average="macro")
print('Downstream Task Accuracy:', micro, macro)
# print('Downstream Task Accuracy:', np.sum(clf.predict(X_test) == y_test)/len(y_test))









