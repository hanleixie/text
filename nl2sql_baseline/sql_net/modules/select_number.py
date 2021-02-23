# -*- coding:utf-8 -*-
# @Time: 2021/2/1 11:01
# @File: select_number.py
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

class SelNumPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, use_ca):
        super(SelNumPredictor, self).__init__()
        self.N_h = N_h
        self.use_ca = use_ca
        self.sel_num_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                                    num_layers=N_depth, batch_first=True,
                                    dropout=0.3, bidirectional=True)#nn.LSTM(300,16,N_h//2)输入的维度为[batch_size, sqe_len, em_size]是batch_size=True，否则[seq_len,batch_size,em_size],输出[seq_len, batch, hidden_size * num_directions]
        self.sel_num_att = nn.Linear(N_h, 1)# 预测select nums 个列，因lstm的输出为[seq_len, batch, hidden_size * num_directions]，故nn.linear中为N_h，因输入的维度[a,b,c],故输出维度为[a,b,1]
        self.sel_num_col_att = nn.Linear(N_h, 1)
        self.sel_num_out = nn.Sequential(nn.Linear(N_h, N_h),
                                         nn.Tanh(), nn.Linear(N_h, 4))# 一个有序的容器，里面的神经网络顺序执行输出最后一维是 4
        self.softmax = nn.Softmax(dim=-1)
        self.sel_num_col2hid1 = nn.Linear(N_h, 2 * N_h)
        self.sel_num_col2hid2 = nn.Linear(N_h, 2 * N_h)


        if self.use_ca:
            print("在预测列数目时使用列注意力机制")
    #x_emb_var:问题的词向量表示，x_len：每个问题的长度，col_inp_var:每个表列名的词向量表示，col_name_len:每个列名的长度，col_len:每个样本几个列
    def forward(self, x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num):
        B = len(x_len)
        max_x_len = max(x_len)
        '''预测所选零件的数量，首先使用列嵌入来计算初始隐藏单元，然后运行LSTM并预测select number'''
        # Predict the number of select part
        # First use column embeddings to calculate the initial hidden unit
        # Then run the LSTM and predict select number
        e_num_col, col_num = col_name_encode(col_inp_var, col_name_len,
                                             col_len, self.sel_num_lstm)#e_numcol:对列进行编码[batch_size,max_len,embedding_size]
        num_col_att_val = self.sel_num_col_att(e_num_col).squeeze(-1)#【16,14】batch=19，每个列有一个得分
        for idx, num in enumerate(col_num):
            if num < max(col_num):
                num_col_att_val[idx, num:] = -1000000#对于那些小与最大值的补齐，给一个特小的值
        num_col_att = self.softmax(num_col_att_val)#softmax层【16，19】
        K_num_col = (e_num_col * num_col_att.unsqueeze(2)).sum(1)#[16,100]#对列编码后乘以它的softmax得分
        sel_num_h1 = self.sel_num_col2hid1(K_num_col).view(B, 4, self.N_h//2).transpose(0, 1).contiguous()#【4,16,50】
        sel_num_h2 = self.sel_num_col2hid2(K_num_col).view(B, 4, self.N_h//2).transpose(0, 1).contiguous()
        #对问题进行编码
        h_num_enc, _ = run_lstm(self.sel_num_lstm, x_emb_var, x_len,
                                hidden=(sel_num_h1, sel_num_h2))#【16,49,100】

        num_att_val = self.sel_num_att(h_num_enc).squeeze(-1)#[batch.size, max_len]=【16,49】
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                num_att_val[idx, num:] = -1000000
        num_att = self.softmax(num_att_val)

        K_sel_num = (h_num_enc * num_att.unsqueeze(2).expand_as(
            h_num_enc)).sum(1)#【16,100】
        sel_num_score = self.sel_num_out(K_sel_num)#【16,4】
        return sel_num_score#对问题的编码







