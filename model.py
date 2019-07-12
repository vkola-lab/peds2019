#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:04:05 2019

@author: Chonghua Xue (Kolachalama's Lab, BU)
"""

from lstm_bi import LSTM_Bi
from utils_data import ProteinSeqDataset, aa2id_i, aa2id_o, collate_fn
from tqdm import tqdm
import numpy as np
import torch
import sys

class ModelLSTM:
    def __init__(self, embedding_dim=64, hidden_dim=64, device='cpu', gapped=True, fixed_len=True):
        self.gapped = gapped
        in_dim, out_dim = len(aa2id_i[gapped]), len(aa2id_o[gapped])
        self.nn = LSTM_Bi(in_dim, embedding_dim, hidden_dim, out_dim, device, fixed_len)
        self.to(device)
        
    def fit(self, trn_fn, vld_fn, n_epoch=100, trn_batch_size=128, vld_batch_size=512, lr=.2, save_fp=None):
        # loss function and optimization algorithm
        loss_fn = torch.nn.NLLLoss()
        op = torch.optim.Adam(self.nn.parameters(), lr=lr)
        
        # to track minimum validation loss
        min_loss = np.inf
        
        # dataset and dataset loader
        trn_data = ProteinSeqDataset(trn_fn, self.gapped)
        vld_data = ProteinSeqDataset(vld_fn, self.gapped)
        if trn_batch_size == -1: trn_batch_size = len(trn_data)
        if vld_batch_size == -1: vld_batch_size = len(vld_data)
        trn_dataloader = torch.utils.data.DataLoader(trn_data, trn_batch_size, True, collate_fn=collate_fn)
        vld_dataloader = torch.utils.data.DataLoader(vld_data, vld_batch_size, False, collate_fn=collate_fn)
        
        for epoch in range(n_epoch):
            # training
            self.nn.train()
            loss_avg, acc_avg, cnt = 0, 0, 0
            with tqdm(total=len(trn_data), desc='Epoch {:03d} (TRN)'.format(epoch), ascii=True, unit='seq', bar_format='{l_bar}{r_bar}') as pbar:
                for batch, batch_flatten in trn_dataloader:
                    # targets
                    batch_flatten = torch.tensor(batch_flatten, device=self.nn.device)
                    
                    # forward and backward routine
                    self.nn.zero_grad()
                    scores = self.nn(batch, aa2id_i[self.gapped])
                    loss = loss_fn(scores, batch_flatten)
                    loss.backward()
                    op.step()
                    
                    # compute statistics
                    L = len(batch_flatten)
                    predicted = torch.argmax(scores, 1)
                    loss_avg = (loss_avg * cnt + loss.data.cpu().numpy() * L) / (cnt + L)
                    corr = (predicted == batch_flatten).data.cpu().numpy()
                    acc_avg = (acc_avg * cnt + sum(corr)) / (cnt + L)
                    cnt += L
                    
                    # update progress bar
                    pbar.set_postfix({'loss': '{:.6f}'.format(loss_avg), 'acc':  '{:.6f}'.format(acc_avg)})
                    pbar.update(len(batch))
            
            # validation
            self.nn.eval()
            loss_avg, acc_avg, cnt = 0, 0, 0
            with torch.set_grad_enabled(False):
                with tqdm(total=len(vld_data), desc='          (VLD)'.format(epoch), ascii=True, unit='seq', bar_format='{l_bar}{r_bar}') as pbar:
                    for batch, batch_flatten in vld_dataloader:
                        # targets
                        batch_flatten = torch.tensor(batch_flatten, device=self.nn.device)
                        
                        # forward routine
                        scores = self.nn(batch, aa2id_i[self.gapped])
                        loss = loss_fn(scores, batch_flatten)
                        
                        # compute statistics
                        L = len(batch_flatten)
                        predicted = torch.argmax(scores, 1)
                        loss_avg = (loss_avg * cnt + loss.data.cpu().numpy() * L) / (cnt + L)
                        corr = (predicted == batch_flatten).data.cpu().numpy()
                        acc_avg = (acc_avg * cnt + sum(corr)) / (cnt + L)
                        cnt += L
                        
                        # update progress bar
                        pbar.set_postfix({'loss': '{:.6f}'.format(loss_avg), 'acc':  '{:.6f}'.format(acc_avg)})
                        pbar.update(len(batch))
            
            # save model
            if loss_avg < min_loss and save_fp:
                min_loss = loss_avg
                self.save('{}/lstm_{:.6f}.npy'.format(save_fp, loss_avg))
    
    def eval(self, fn, batch_size=512):        
        # dataset and dataset loader
        data = ProteinSeqDataset(fn, self.gapped)
        if batch_size == -1: batch_size = len(data)
        dataloader = torch.utils.data.DataLoader(data, batch_size, True, collate_fn=collate_fn)
        
        self.nn.eval()
        scores = np.zeros(len(data), dtype=np.float32)
        sys.stdout.flush()
        with torch.set_grad_enabled(False):
            with tqdm(total=len(data), ascii=True, unit='seq', bar_format='{l_bar}{r_bar}') as pbar:
                for n, (batch, batch_flatten) in enumerate(dataloader):
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
                    scores[n*batch_size:(n+1)*batch_size] = batch_scores
                    pbar.update(len(batch))
        return scores
    
    def save(self, fn):
        param_dict = self.nn.get_param()
        param_dict['gapped'] = self.gapped
        np.save(fn, param_dict)
    
    def load(self, fn):
        param_dict = np.load(fn).item()
        self.gapped = param_dict['gapped']
        self.nn.set_param(param_dict)

    def to(self, device):
        self.nn.to(device)
        self.nn.device = device
        
    def summary(self):
        for n, w in self.nn.named_parameters():
            print('{}:\t{}'.format(n, w.shape))
#        print('LSTM: \t{}'.format(self.nn.lstm_f.all_weights))
        print('Fixed Length:\t{}'.format(self.nn.fixed_len) )
        print('Gapped:\t{}'.format(self.gapped))
        print('Device:\t{}'.format(self.nn.device))
            