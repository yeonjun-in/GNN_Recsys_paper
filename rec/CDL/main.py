import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import os, sys

from dataset import add_noise, CDLDataset
from model import CDL, SDAE

os.chdir('/home/yeonjun/Desktop/RecSys_implementation/CDL')
sys.path.append('/home/yeonjun/Desktop/RecSys_implementation/CDL')

class Config:
    learning_rate = 0.001
    early_stopping_round = 0
    epochs = 15
    seed = 1995
    dim_f = 10
    batch_size = 16
    lambda_u = 1
    lambda_w = 5e-4
    lambda_v = 1
    lambda_n = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rankM = 50

    
config = Config()

train = np.load('./data/ml_100k_train.npy')
train = np.where(train > 0, 1, 0)
test = np.load('./data/ml_100k_test.npy')
test = np.where(test > 0, 1, 0)

xc = pd.read_csv('./data/movies.csv').iloc[:,5:].values
x0 = add_noise(xc, corrupt_ratio=0.1)

config.n_item = train.shape[1]
config.n_user = train.shape[0]
idx = np.arange(config.n_item)
config.neg_item_tr = {i :idx[~train[i, :].astype(bool)] for i in range(config.n_user)}
config.neg_item_tst = {i :idx[~test[i, :].astype(bool)] for i in range(config.n_user)}

config.pos_item_tr_bool = {i :train[i, :].astype(bool) for i in range(config.n_user)}
config.pos_item_tst_bool = {i :test[i, :].astype(bool) for i in range(config.n_user)}



dataset = CDLDataset(xc, x0)
trainloader = DataLoader(dataset, config.batch_size, drop_last=False, shuffle=False)
model = CDL(
    train_imp=train, 
    test_imp=test, 
    input_dim=xc.shape[1], 
    hidden_dim=config.dim_f, 
    dim_f=config.dim_f,
    dataloader=trainloader,
    seed=1995,
    device=config.device,
    config=config
)

model.fit()