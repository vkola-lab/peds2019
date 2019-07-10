#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:04:05 2019

@author: cxue2
"""

from lstm_bi import LSTM_Bi
from utils_data import ProteinSeqDataset, aa2id_i, aa2id_o, collate_fn
from tqdm import tqdm, tqdm_notebook
import numpy as np
import torch

class ModelLSTM:
    def __init__(self, embedding_dim, hidden_dim, device='cpu', gapped=True, fixed_len=True):
        self.gapped = gapped
        in_dim, out_dim = len(aa2id_i[gapped]), len(aa2id_o[gapped])
        self.nn = LSTM_Bi(in_dim, embedding_dim, hidden_dim, out_dim, device, fixed_len)
        self.to(device)
        self.tqdm = tqdm
        
    def fit(self, trn_fn, vld_fn, n_epoch=100, trn_batch_size=32, vld_batch_size=-1, lr=.2, save_model=None, n_iter_per_print=100):
        # loss function and optimization algorithm
        loss_fn = torch.nn.NLLLoss()
        op = torch.optim.Adam(self.nn.parameters(), lr=lr)
        
        # to track minimum validation loss
        min_val_loss = np.inf
        
        # dataset and dataset loader
        trn_data = ProteinSeqDataset(trn_fn, self.gapped)
        vld_data = ProteinSeqDataset(vld_fn, self.gapped)
        if trn_batch_size == -1: trn_batch_size = len(trn_data)
        if vld_batch_size == -1: vld_batch_size = len(vld_data)
        trn_dataloader = torch.utils.data.DataLoader(trn_data, trn_batch_size, True, collate_fn=collate_fn)
        vld_dataloader = torch.utils.data.DataLoader(vld_data, vld_batch_size, False, collate_fn=collate_fn)
        
        for epoch in range(n_epoch):
            print('epoch: {}'.format(epoch))
            
            # training
            self.nn.train()
            loss_all, corr_all, count = 0, 0, 0
            
            for _iter, (batch, batch_flatten) in enumerate(self.tqdm(trn_dataloader)):
                # targets
                batch_flatten = torch.tensor(batch_flatten, device=self.nn.device)
                
                # forward and backward routine
                self.nn.zero_grad()
                scores = self.nn(batch, aa2id_i[self.gapped])
                loss = loss_fn(scores, batch_flatten)
                loss.backward()
                op.step()
                
                # compute statistics
                count += len(batch_flatten)
                predicted = torch.argmax(scores, 1)
                loss_all += loss.data.cpu().numpy() * len(batch_flatten)           # total loss
                corr_all += (predicted == batch_flatten).sum().data.cpu().numpy()  # totol hits
                
                # print statistics
                if _iter % n_iter_per_print == 0 and _iter != 0:
                    tqdm.write('\ttrain_loss: {}\ttrain_acc: {}'.format(loss_all / count, corr_all / count))
                    loss_all, corr_all, count = 0, 0, 0
            
            # validation
            self.nn.eval()
            loss_all, corr_all, count = 0, 0, 0
            with torch.set_grad_enabled(False):
                for _iter, (batch, batch_flatten) in enumerate(self.tqdm(vld_dataloader)):
                    batch_flatten = torch.tensor(batch_flatten, device=self.nn.device)
                    scores = self.nn(batch, aa2id_i[self.gapped]) 
                    count += len(batch_flatten)
                    loss = loss_fn(scores, batch_flatten)
                    predicted = torch.argmax(scores, 1)
                    loss_all += loss.data.cpu().numpy() * len(batch_flatten)
                    corr_all += (predicted == batch_flatten).sum().data.cpu().numpy()  
            print('val_loss: {}\tval_acc: {}'.format(loss_all / count, corr_all / count))
            
            # save model
            val_loss = loss_all / count
            if val_loss < min_val_loss and save_model:
                min_val_loss = val_loss
                self.save_model('./model/temp/lstm_{}.npy'.format(val_loss))
    
    def eval(self, fn, batch_size=-1):
        # dataset and dataset loader
        data = ProteinSeqDataset(fn, self.gapped)
        if batch_size == -1: batch_size = len(data)
        dataloader = torch.utils.data.DataLoader(data, batch_size, True, collate_fn=collate_fn)
        
        self.nn.eval()
        scores = np.zeros(len(data), dtype=np.float32)
        with torch.set_grad_enabled(False):
            for n, (batch, batch_flatten) in enumerate(self.tqdm(dataloader)):
                actual_batch_size = len(batch)  # last iteration may contain less sequences
                seq_len = [len(seq) for seq in batch]
                seq_len_cumsum = np.cumsum(seq_len)
                out = self.nn(batch, aa2id_i[self.gapped]).data.cpu().numpy()
                out = np.split(out, seq_len_cumsum)[:-1]
                batch_scores = []
                for i in range(actual_batch_size):
                    pos_scores = []
                    for j in range(seq_len[i]):
                        pos_scores.append(out[i][j, batch[i][j]])
                    batch_scores.append(-sum(pos_scores) / seq_len[i])    
                scores[n*batch_size:n*batch_size+actual_batch_size] = batch_scores
        return scores
    
    def save_model(self, fn):
        param_dict = self.nn.get_param()
        np.save(fn, param_dict)
    
    def load_model(self, fn):
        param_dict = np.load(fn).item()
        self.nn.set_param(param_dict)

    def to(self, device):
        self.nn.to(device)
        self.nn.device = device
        
    def _tqdm_mode(self, mode):
        exec('self.tqdm = {}'.format(mode))