# -*- coding: utf-8 -*-
# @Time     : 2022/2/17 14:55
# @Author   : xiehl
# @Software : PyCharm
# ---------------------------
import numpy as np
import pickle
import torch
from . import tools


class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.load_data(mmap)

    def load_data(self, mmap):

        # data: N C V T M
        """
        train: {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 19, 20, 21}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9750992/9750992 [00:11<00:00, 862844.50it/s]
3010669
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2437748/2437748 [00:10<00:00, 229586.52it/s]
valid: {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 19, 20, 21}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2437748/2437748 [00:02<00:00, 876122.20it/s]
752722
        Args:
            mmap:

        Returns:

        """
        # load label
        with open(self.label_path, 'rb') as f:
            # self.sample_name, self.label = pickle.load(f)
            self.label = pickle.load(f)#[:200000, :]# 3611888



        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')#[:200000, :, :]
        else:
            self.data = np.load(self.data_path)#[:200000, :, :]
        if self.debug:
            self.label = self.label[0:10000, :]
            self.data = self.data[0:10000, :, :]
            # self.sample_name = self.sample_name[0:1000]

        """
        N = 4 多少个故障小区
        C = 3 (X,Y,S)代表一个点的信息(位置+预测的可能性)# 未理解
        T = 24 一个小区在时间周期内有多少个时间段
        V = 254 一共有多少个节点
        M = 1 人数，paper中将人数限定在最大2个人
        """

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # # processing
        # if self.random_choose:
        #     data_numpy = tools.random_choose(data_numpy, self.window_size)
        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
        # if self.random_move:
        #     data_numpy = tools.random_move(data_numpy)

        return data_numpy, label