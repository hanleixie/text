# -*- coding:utf-8 -*-
# @Time: 2021/2/1 10:58
# @File: aggregator_predict.py
# @Software: PyCharm
# @Author: xiehl
# -------------------------
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm, col_name_encode



class AggPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, use_ca):
        super(AggPredictor, self).__init__()
        self.use_ca = use_ca

        self.agg_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        if use_ca:
            print("在预测列数目时使用列注意力机制")
            self.agg_col_name_enc = nn.LSTM(input_size=N_word,
                    hidden_size=N_h//2, num_layers=N_depth,
                    batch_first=True, dropout=0.3, bidirectional=True)
            self.agg_att = nn.Linear(N_h, N_h)
        else:
            print("在预测列数目时不使用列注意力机制")
            self.agg_att = nn.Linear(N_h, 1)
        self.agg_out = nn.Sequential(nn.Linear(N_h, N_h), nn.Tanh(), nn.Linear(N_h, 6))
        self.softmax = nn.Softmax(dim=-1)
        self.agg_out_K = nn.Linear(N_h, N_h)
        self.col_out_col = nn.Linear(N_h, N_h)

    def forward(self, x_emb_var, x_len, col_inp_var=None, col_name_len=None,
            col_len=None, col_num=None, gt_sel=None, gt_sel_num=None):
        B = len(x_emb_var)
        max_x_len = max(x_len)

        e_col, _ = col_name_encode(col_inp_var, col_name_len, col_len, self.agg_col_name_enc)#【16,19,100】
        h_enc, _ = run_lstm(self.agg_lstm, x_emb_var, x_len)#[16,49,100]

        col_emb = []
        for b in range(B):
            cur_col_emb = torch.stack([e_col[b,x] for x in gt_sel[b]] + [e_col[b,0]] * (4-len(gt_sel[b])))
            col_emb.append(cur_col_emb)
        col_emb = torch.stack(col_emb)#[16,4,100]

        att_val = torch.matmul(self.agg_att(h_enc).unsqueeze(1), col_emb.unsqueeze(3)).squeeze()#[16,4,49]

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                att_val[idx, num:] = -100
        att = self.softmax(att_val.view(B*4, -1)).view(B, 4, -1)#[16,4,49]

        K_agg = (h_enc.unsqueeze(1) * att.unsqueeze(3)).sum(2)
        agg_score = self.agg_out(self.agg_out_K(K_agg) + self.col_out_col(col_emb))#.squeeze()为了单个样本注释掉后面的.squeeze()
        return agg_score#[16,4,6]
