#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:04:05 2019

@author: cxue2
"""

import sys
sys.path.insert(0, '..') if '..' not in sys.path else 0
from lstm_bi import LSTM_Bi
from util.rand_batch_gen import rand_batch_gen
import util.PDB_info_v3 as pdbinfo
import numpy as np
import torch
from tqdm import tqdm
from math import ceil

class ModelLSTM:
    def __init__(self, embedding_dim, hidden_dim, vocab_dict, device='cpu'):
        self.vocab_dict = vocab_dict
        self.nn = LSTM_Bi(embedding_dim, hidden_dim, len(vocab_dict), device)
        self.to(device)
        
    def fit(self, Xt, Xv, n_epoch=100, b_size=32, lr=.2, save_model=False, n_iter_per_print=100):
        loss_fn = torch.nn.NLLLoss()
        op = torch.optim.Adam(self.nn.parameters(), lr=lr)        
        min_val_loss = np.inf
        for epoch in range(n_epoch):
            print('epoch: ' + str(epoch))
            
            # training
            self.nn.train()
            all_train_batch_idx = rand_batch_gen(b_size, len(Xt))
            loss_all = 0
            corr_all = 0
            count = 0
            
            for _iter, batch_idx in enumerate(tqdm(all_train_batch_idx)):
                X_batch = [Xt[idx] for idx in batch_idx]
                
                # sorting
                X_batch = sorted(X_batch, key=lambda p: len(p))[::-1]

                Y_batch = [x for seq in X_batch for x in seq]
                Y_batch = torch.tensor(Y_batch).to(self.nn.device)
                
                self.nn.zero_grad()
                scores = self.nn(X_batch, pdbinfo.aa2id)
                loss = loss_fn(scores, Y_batch)
                loss.backward()
                op.step()
                
                count += len(Y_batch)
                predicted = torch.argmax(scores, 1)
                loss_all += loss.data.cpu().numpy() * len(Y_batch)  # total loss
                corr_all += (predicted == Y_batch).sum().data.cpu().numpy()  # totol hits
                
                if _iter % n_iter_per_print == 0 and _iter != 0:
                    tqdm.write('\ttrain_loss: ' + str(loss_all / count) +
                               '\ttrain_acc: ' + str(corr_all / count))
                    loss_all = 0
                    corr_all = 0
                    count = 0
            
            # validation
            self.nn.eval()
            all_batch_idx = rand_batch_gen(b_size, len(Xv))
            loss_all = 0
            corr_all = 0
            count = 0
            with torch.set_grad_enabled(False):
                for _iter, batch_idx in enumerate(all_batch_idx):
                    X_batch = [Xv[idx] for idx in batch_idx]
                    X_batch = sorted(X_batch, key=lambda p: len(p))[::-1]
                    Y_batch = [x for seq in X_batch for x in seq]
                    Y_batch = torch.tensor(Y_batch).to(self.nn.device)
                    scores = self.nn(X_batch, pdbinfo.aa2id) 
                    count += len(Y_batch)
                    loss = loss_fn(scores, Y_batch)
                    predicted = torch.argmax(scores, 1)
                    loss_all += loss.data.cpu().numpy() * len(Y_batch)
                    corr_all += (predicted == Y_batch).sum().data.cpu().numpy()  
            print('val_loss: ' + str(loss_all/count) + 
                  '\tval_acc: ' + str(corr_all/count))
            
            # save model
            val_loss = loss_all / count
            if val_loss < min_val_loss and save_model:
                min_val_loss = val_loss
                self.save_model('./model/temp/lstm_' + str(val_loss) + '.npy')
    
    def score(self, X, b_size=32):
        scores = np.zeros(len(X), dtype=np.float32)
        n_seg = ceil(len(X) / b_size)
        L = len(X[0])
        self.nn.eval()
        with torch.set_grad_enabled(False):
            for n in tqdm(range(n_seg)):
                batch = X[n*b_size:(n+1)*b_size]
                batch = sorted(batch, key=lambda p: len(p))[::-1]
                b_len = [len(b) for b in batch]
                n_seq = len(batch)
                out = self.nn(batch, self.vocab_dict)
                out_np = out.data.cpu().numpy()
                b_len_cumsum = np.cumsum(b_len)
                out_np = np.split(out_np, b_len_cumsum)[:-1]
                out_np = [b.reshape((-1, len(self.vocab_dict))) for b in out_np]
#                out_np = out.data.cpu().numpy().reshape((n_seq,L,len(self.vocab_dict)))
                scores[n*b_size:(n+1)*b_size] = [-sum([out_np[i][j, batch[i][j]] for j in range(b_len[i])]) / b_len[i] for i in range(n_seq)]
        return scores
    
    def save_model(self, fp):
        param_dict = self.nn.get_param()
        np.save(fp, param_dict)
    
    def load_model(self, fp):
        param_dict = np.load(fp).item()
        self.nn.set_param(param_dict)

    def to(self, device):
        if device not in ['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']:
            raise Exception('Invalid device.')
        self.nn.to(device)
        self.nn.device = device
