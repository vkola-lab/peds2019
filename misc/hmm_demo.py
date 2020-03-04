#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:08:17 2020

@author: cxue2
"""

""" This is a demo script for using HMM to score antibody sequences.
hmmlearn is an HMM library written in Python. It can be downloaded via pip.
- hmm_demo.py """

import itertools
from hmmlearn import hmm
import sys
sys.path.append('..')
from utils_data import vocab_o

# load training data
print('Loading data for training...', end=' ')
with open('../data/sample/human_train.txt', 'r') as f:
    X_trn = [l.strip('\n') for l in f]
    
# convert amino acid letter to integer id
aa2id = dict(zip(vocab_o[True],  list(range(len(vocab_o[True])))))
X_trn = [[[aa2id[ltr]] for ltr in seq] for seq in X_trn]
print('Done.', flush=True)

# train on a subset for illustration purpose
# comment the line if the user wants to train on the entire dataset
X_trn = X_trn[:10]

# flatten X and record lengths
X_trn_len = [len(seq) for seq in X_trn]
X_trn = list(itertools.chain(*X_trn))

# initialize and train HMM model
print('Fitting model...', end='\n')
model = hmm.MultinomialHMM(n_components=128, n_iter=100, verbose=True)
model.fit(X_trn, lengths=X_trn_len)
print('Done.', flush=True)

# load testing data and conduct necessary pre-processing
print('Loading data for evaluating...', end=' ')
with open('../data/sample/human_test.txt', 'r') as f:
    X_tst = [l.strip('\n') for l in f]
X_tst = [[[aa2id[ltr]] for ltr in seq] for seq in X_tst]
print('Done.', flush=True)

# comment the line if the user wants to evaluate on the entire dataset
X_tst = X_tst[:10]

# evaluate testing sequences
print('Evaluating testing sequences...', end=' ')
scores = [-model.score(seq) / len(seq) for seq in X_tst]
print('Done.', flush=True)