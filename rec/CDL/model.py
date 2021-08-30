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

class SDAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim):
        super(SDAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc3 = nn.Linear(hidden_dim, embed_dim)
        
        self.dec1 = nn.Linear(embed_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        latent = F.relu(self.enc3(x))
        x = F.relu(self.dec1(latent))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return latent, x

class CDL:
    
    def __init__(self, train_imp, test_imp, input_dim, hidden_dim, dim_f, dataloader, seed, device, config):
        self.dim_f = dim_f
        self.user_num = train_imp.shape[0]
        self.item_num = train_imp.shape[1]
        self.input_dim = input_dim

        self.R_tr = train_imp
        self.R_tst = test_imp
        self.C = np.where(self.R_tr > 0, 1, 0)
        self.C_u = np.zeros((self.item_num, self.item_num))
        self.C_i = np.zeros((self.user_num, self.user_num))

        np.random.seed(seed)
        self.X = np.random.standard_normal((self.user_num, dim_f))
        self.Y = np.random.standard_normal((self.item_num, dim_f))
        
        self.loss_tr = defaultdict(float)
        self.loss_ae = defaultdict(float)
        self.loss_tst = defaultdict(float)

        self.ae = SDAE(input_dim=input_dim, hidden_dim=hidden_dim, embed_dim=dim_f).to(device)
        self.optimizer = optim.Adam(self.ae.parameters(), lr=config.learning_rate, weight_decay=config.lambda_w)
        self.dataloader = dataloader

        self.lambda_u = config.lambda_u
        self.lambda_w = config.lambda_w
        self.lambda_v = config.lambda_v
        self.lambda_n = config.lambda_n

        self.device = device
        self.config = config
    
    def ae_train(self):
        latent_np = np.zeros((self.item_num, self.dim_f))
        loss_ae = []
        for batch in self.dataloader:
            y = batch['clean'].to(self.device)
            x = batch['corrupt'].to(self.device)
            idx = batch['idx']
            latent, pred = self.ae(x)
            latent_ = latent.detach().cpu().numpy()
            latent_np[idx.numpy()] = latent_

            loss = self.loss_fn(pred, y, idx.to(self.device), latent_)
            loss.backward()
            self.optimizer.step()
            loss_ae.append(loss.item())

        return latent_np, np.mean(loss_ae)

    def fit(self):
        start = datetime.now()
        for epoch in range(self.config.epochs):
            start_epoch = datetime.now()
            self.ae.train()
            self.latent_feat, self.loss_ae[epoch] = self.ae_train()
            n = 0
            for u in range(self.user_num):
                yty = np.dot(self.Y.T, self.Y)
                self.X[u, :] = self.update_user_vector(u, yty)

            for i in range(self.item_num):
                xtx = np.dot(self.X.T, self.X)
                self.Y[i, :] = self.update_item_vector(i, xtx)
            
            phat = self.scoring()
            
            train_loss = self.evaluate(train_eval=True)
            test_loss = self.evaluate(train_eval=False)

            self.loss_tr[epoch] = train_loss
            self.loss_tst[epoch] = test_loss


            print(f'EPOCH {epoch+1} : TRAINING RANK {self.loss_tr[epoch]:.5f}, VALID RANK {self.loss_tst[epoch]:.5f}')

            print(f'Time per one epoch {datetime.now() - start_epoch}')
        end = datetime.now()
        print(f'Training takes time {end-start}')
        
    def scoring(self):
        
        return np.dot(self.X, self.Y.T)
    
    def update_user_vector(self, u, yty):
        np.fill_diagonal(self.C_u, (self.C[u, :] - 1))
        comp1 = yty
        comp2 = np.dot(self.Y.T, self.C_u).dot(self.Y)
        comp3 = np.identity(self.config.dim_f) * self.config.lambda_u
        comp = np.linalg.inv(comp1 + comp2 + comp3)
        self.C_u = self.C_u + np.identity(self.C_u.shape[0])
        comp = np.dot(comp, self.Y.T).dot(self.C_u)
        
        return np.dot(comp, self.R_tr[u, :])

    def update_item_vector(self, i, xtx): 
        np.fill_diagonal(self.C_i, (self.C[:, i] - 1))
        comp1 = xtx
        comp2 = np.dot(self.X.T, self.C_i).dot(self.X)
        comp3 = np.identity(self.config.dim_f) * self.config.lambda_v
        comp = np.linalg.inv(comp1 + comp2 + comp3)
        self.C_i = self.C_i + np.identity(self.C_i.shape[0])
        comp4 = self.X.T.dot(self.C_i).dot(self.R_tr[:, i])
        comp5 = self.lambda_v * self.latent_feat[i, :]
        
        return np.dot(comp, comp4+comp5)
    
    def loss_fn(self, pred, xc, idx, latent_feat):
        X = torch.tensor(self.X).to(self.device)
        Y = torch.tensor(self.Y).to(self.device)[idx, :]
        R = torch.tensor(self.R_tr).float().to(self.device)[:, idx]
        C = torch.tensor(self.C).float().to(self.device)[:, idx]
        latent = torch.tensor(latent_feat).to(self.device)

        comp1 = (X**2).sum(axis=1).sum() * self.lambda_u/2
        comp2 = ((Y - latent)**2).sum(axis=1).sum() * self.lambda_v/2
        comp3 = ((pred - xc)**2).sum(axis=1).sum() * self.lambda_n/2
        comp4 = torch.sum((torch.mm(X, Y.T) - R)**2 * C/2)

        return comp1+comp2+comp3+comp4

    def evaluate(self, train_eval):
        if train_eval:
            R = self.R_tr
        else:
            R = self.R_tst

        phat = self.scoring()
        rank_mat = np.zeros(phat.shape)
        for u in range(self.user_num):
            pred_u = phat[u, :] * -1
            rank = pred_u.argsort().argsort()
            rank = rank / self.item_num
            rank_mat[u, :] = rank

        return np.sum(R * rank_mat) / np.sum(R)