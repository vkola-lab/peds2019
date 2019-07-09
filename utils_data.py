#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:05:54 2019

@author: cxue2
"""

class ProteinSeqDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, fn):
        self.

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        X = np.load('{}/{}'.format(self.root, self.fns[idx]))
        y = self.labels[idx]
        return X, y
    
    def to_stage(self, stage):
        assert stage in ['train', 'valid']
        self.stage = stage