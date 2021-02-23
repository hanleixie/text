# -*- coding:utf-8 -*-
# @Time: 2021/2/1 11:01
# @File: net_utils.py
# @Software: PyCharm
# @Author: xiehl
# -------------------------
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

def run_lstm(lstm, inp, inp_len, hidden=None):
    '''使用压缩序列运行LSTM。'''
    # This requires to first sort the input according to its length.
    sort_perm = np.array(sorted(range(len(inp_len)),
        key=lambda k:inp_len[k], reverse=True))#对所在的位置索引反转排序输出
    sort_inp_len = inp_len[sort_perm]#对长度进行排序逆向输出
    sort_perm_inv = np.argsort(sort_perm)#索引排序
    if inp.is_cuda:
        sort_perm = torch.LongTensor(sort_perm).cuda()
        sort_perm_inv = torch.LongTensor(sort_perm_inv).cuda()

    lstm_inp = nn.utils.rnn.pack_padded_sequence(inp[sort_perm],#将变长序列压缩，输入时[batch.size, max_len, embedding.size],返回时打包成一条的既去掉为了和最大长度相同而拼凑的0
            sort_inp_len, batch_first=True)
    if hidden is None:
        lstm_hidden = None
    else:
        lstm_hidden = (hidden[0][:, sort_perm], hidden[1][:, sort_perm])

    sort_ret_s, sort_ret_h = lstm(lstm_inp, lstm_hidden)
    ret_s = nn.utils.rnn.pad_packed_sequence(
            sort_ret_s, batch_first=True)[0][sort_perm_inv]#将pack压缩后的重新填充，[列_num, max.len, 100]
    ret_h = (sort_ret_h[0][:, sort_perm_inv], sort_ret_h[1][:, sort_perm_inv])
    return ret_s, ret_h


def col_name_encode(name_inp_var, name_len, col_len, enc_lstm):
    '''对列进行编码，列名的嵌入是其LSTM输出的最后一个状态。'''
    #The embedding of a column name is the last state of its LSTM output.
    name_hidden, _ = run_lstm(enc_lstm, name_inp_var, name_len)
    name_out = name_hidden[tuple(range(len(name_len))), name_len-1]#每一个
    ret = torch.FloatTensor(
            len(col_len), max(col_len), name_out.size()[1]).zero_()#[16,16,100]=[batch_size,max_len,embedding_size]
    if name_out.is_cuda:
        ret = ret.cuda()

    st = 0
    for idx, cur_len in enumerate(col_len):
        ret[idx, :cur_len] = name_out.data[st:st+cur_len]
        st += cur_len
    ret_var = Variable(ret)

    return ret_var, col_len
