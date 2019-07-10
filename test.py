# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 21:28:33 2019

@author: Iluva
"""

from model import ModelLSTM

# initialize model
model = ModelLSTM(embedding_dim=64, hidden_dim=64, device='cuda:0', gapped=True, fixed_len=True)
print('Model initialized.')

# data file names
trn_fn = './data/sample/human_train.txt'
vld_fn = './data/sample/human_val.txt'

# fit model
model.fit(trn_fn=trn_fn, vld_fn=vld_fn, n_epoch=2, trn_batch_size=1024, vld_batch_size=1024, lr=.002,
          save_model=None, n_iter_per_print=100)