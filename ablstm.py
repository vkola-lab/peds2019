# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:02:40 2019

@author: Chonghua Xue (Kolachalama's Lab, BU)
"""

import argparse
from utils_miscellany import load_config_xml
from model import ModelLSTM

if __name__ == '__main__':
    # main parser
    parser = argparse.ArgumentParser(description='Quantifying the nativeness of antibody sequences using long short-term memory network.')
    subparsers = parser.add_subparsers(dest='cmd')
    
    # fit cmd parser
    parser_fit = subparsers.add_parser('fit')
    parser_fit.add_argument('TRN_FN', help='training data file')
    parser_fit.add_argument('VLD_FN', help='validation data file')
    parser_fit.add_argument('SAVE_FP', help='model save path')
    parser_fit.add_argument('-l', default='', help='model file to load (default: \"\")')
    parser_fit.add_argument('-c', default='./ablstm.config', help='configuration XML file (default: \"./ablstm.config\")')
    parser_fit.add_argument('-d', default='cpu', help='device (default: \"cpu\")')
    
    # eval cmd parser
    parser_eval = subparsers.add_parser('eval')
    parser_eval.add_argument('TST_FN', help='evaluation data file')
    parser_eval.add_argument('MDL_FN', help='model file to load')
    parser_eval.add_argument('SCR_FN', help='file to save scores')
    parser_eval.add_argument('-c', default='./ablstm.config', help='configuration XML file (default: \"./ablstm.config\")')
    parser_eval.add_argument('-d', default='cpu', help='device (default: \"cpu\")')
    
    # args is stored in Namespace obj and configuration in dict
    args = parser.parse_args()
    conf = load_config_xml(args.c)
    
    if args.cmd == 'fit':
        param_fit = {'trn_fn': args.TRN_FN,
                     'vld_fn': args.VLD_FN,
                     'n_epoch': conf['fit']['n_epoch'],
                     'trn_batch_size': conf['fit']['trn_batch_size'],
                     'vld_batch_size': conf['fit']['vld_batch_size'],
                     'lr': conf['fit']['lr'],
                     'save_fp': args.SAVE_FP}
        if not args.l:
            param_init = {'embedding_dim': conf['__init__']['embedding_dim'],
                          'hidden_dim': conf['__init__']['hidden_dim'],
                          'gapped': conf['__init__']['gapped'] == 'True',
                          'fixed_len': conf['__init__']['fixed_len'] == 'True',
                          'device': args.d}
            model = ModelLSTM(**param_init)
        else:
            param_ini = {'device': args.d}
            model = ModelLSTM(**param_ini)
            model.load(args.l)
        model.fit(**param_fit)
        
    elif args.cmd == 'eval':
        param_ini = {'device': args.d}
        model = ModelLSTM(**param_ini)
        model.load(args.MDL_FN)
        param_eval = {'fn': args.TST_FN, 'batch_size': conf['eval']['batch_size']}
        scores = model.eval(**param_eval)
        scores = [str(s) for s in scores]
        with open(args.SCR_FN, 'w') as f:
            f.write(','.join(scores))
