import numpy as np
import pandas as pd
import os, sys
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

from dataset import AutoRecData
from model import AutoRec

os.chdir('/home/yeonjun/Desktop/RecSys_implementation/AutoRec')
sys.path.append('/home/yeonjun/Desktop/RecSys_implementation/AutoRec')

train = np.load('./data/ml_100k_train.npy')
test = np.load('./data/ml_100k_test.npy')

class Config:
    lr = 0.01
    weight_decay = 5e-4
    based_on = 'item'
    batch_size = 64
    input_dim = train.shape[0] if based_on == 'item' else train.shape[1]
    hidden_dim = 15
    epochs = 30
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

trainset = AutoRecData(train, config.based_on)
testset = AutoRecData(test, config.based_on)
trainloader = DataLoader(trainset, batch_size=config.batch_size, shuffle=False, drop_last=False)
testloader = DataLoader(testset, batch_size=config.batch_size*100, shuffle=False, drop_last=False)

model = AutoRec(input_dim=config.input_dim, hidden_dim=config.hidden_dim, output_dim=config.input_dim)
model = model.to(config.device)
optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

history = defaultdict(list)
for epoch in range(config.epochs):
    model.train()
    losses = []
    for x in trainloader:
        optimizer.zero_grad()
        x = x.to(config.device)
        mask = x > 0
        pred = model(x)
        loss = torch.mean(((x - pred)[mask])**2)
        loss.backward()
        optimizer.step()
        losses.append(np.sqrt(loss.item()))
    history['tr'].append(np.mean(losses))

    model.eval()
    with torch.no_grad():
        for x in testloader:
            x = x.to(config.device)
            mask = x > 0
            pred = model(x)
            loss = torch.sqrt(torch.mean(((x - pred)[mask])**2))
            losses.append(loss.item())
    history['test'].append(np.mean(losses))
    print(f'EPOCH {epoch+1}: TRAINING loss {history["tr"][-1]} VALID loss {history["test"][-1]}')

    