# -*- coding:utf-8 -*-
# @Time: 2021/2/22 20:52
# @File: config.py.py
# @Software: PyCharm
# @Author: xiehl
# -------------------------

server_port = 8201

n_word = 300
learning_rate = 1e-3

model_path = 'saved_model/best_model.pt'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=16, help='Batch size')
# parser.add_argument('--epoch', type=int, default=2, help='Epoch number')
parser.add_argument('--gpu', action='store_true', help='训练是否使用gpu')
parser.add_argument('--toy', action='store_true', help='If set, use small data for fast debugging')
parser.add_argument('--ca', action='store_true', help='是否使用列注意')
parser.add_argument('--train_emb', action='store_true', help='Train word embedding for SQLNet')
parser.add_argument('--restore', action='store_true', help='Whether restore trained model')
parser.add_argument('--logdir', type=str, default='', help='Path of save experiment logs')
args = parser.parse_args()