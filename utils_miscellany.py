#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:24:51 2019

@author: Chonghua Xue (Kolachalama's Lab, BU)
"""

import xml.etree.ElementTree as ET

def load_config_xml(fn):
    tree = ET.parse(fn)
    root = tree.getroot()
    rslt = {}
    for c1 in root:
        rslt[c1.tag] = {}
        for c2 in c1:
            rslt[c1.tag][c2.tag] = c2.text
    rslt['__init__']['embedding_dim'] = int(rslt['__init__']['embedding_dim'])
    rslt['__init__']['hidden_dim'] = int(rslt['__init__']['hidden_dim'])
    rslt['__init__']['gapped'] = bool(rslt['__init__']['gapped'])
    rslt['__init__']['fixed_len'] = bool(rslt['__init__']['fixed_len'])
    rslt['fit']['n_epoch'] = int(rslt['fit']['n_epoch'])
    rslt['fit']['trn_batch_size'] = int(rslt['fit']['trn_batch_size'])
    rslt['fit']['vld_batch_size'] = int(rslt['fit']['vld_batch_size'])
    rslt['fit']['lr'] = float(rslt['fit']['lr'])
    rslt['eval']['batch_size'] = int(rslt['eval']['batch_size'])
    return rslt

def to_vlen(src_fn, tar_fn):
    with open(src_fn, 'r') as f:
        lns = [ln.strip('\n').replace('-', '') for ln in f]
    with open(tar_fn, 'w') as f:
        f.write('\n'.join(lns))