# -*- coding: utf-8 -*-
# @Time     : 2022/2/17 14:03
# @Author   : xiehl
# @Software : PyCharm
# ---------------------------
# !/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np
from sklearn.metrics import classification_report

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class
# torch.backends.cudnn.enabled = False
from .processor import Processor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                    0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr
    base = 0
    def show_topk(self, k):
        print("self result shape", self.result.shape)
        rank = self.result.argsort()
        print("rank shape", rank.shape)
        rank_ = []
        # with open(rank_label_path, "r") as f:
        for line in rank:
            last = line[-1]
            rank_.append(float(last))
        self.io.print_log("rank: {}".format(set(rank_)))
        rank_label = self.label.argsort()
        rank_label_ = []
        for line in rank_label:
            last = line[-1]
            rank_label_.append(float(last))
        self.io.print_log("label: {}".format(set(rank_label_)))
        def judge(x):
            matrix = [0.0 for _ in range(2)]
            matrix[x] = 1.0
            return np.array(matrix)

        hit_top_k = [(l == judge(rank[i, -k:][-1])).all() for i, l in enumerate(self.label)]

        # print("hit top k shape", hit_top_k)
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        # 保存最好的结果模型

        if accuracy >= 0.86:
            filename = 'best_model_{}.pt'.format(accuracy)
            self.io.save_model(self.model, filename)
            np.savetxt(r"./data/real_data/rank.txt", rank, fmt='%f',
                       delimiter=',')
            np.savetxt(r"./data/real_data/rank_label.txt", rank_label, fmt='%f',
                       delimiter=',')




        def print_report(rank, rank_label):
            rank_ = []
            for line in rank:
                last = line[-1]
                rank_.append(float(last))

            rank_label_ = []
            for line in rank_label:
                last = line[-1]
                rank_label_.append(float(last))
            target_names = [str(x) for x in list(set(rank_label_))]
            return classification_report(rank_label_, rank_, target_names=target_names)


        self.io.print_log('\tAccury{}: {:.2f}%'.format(k, 100 * accuracy))

        self.io.print_log('\treport is {}%'.format(print_report(rank, rank_label)))


    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label in loader:
            # get data
            data = data.float().to(self.dev)
            # label = label.long().to(self.dev)
            label = label.float().to(self.dev)

            # forward
            output = self.model(data)

            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:

            # get data
            data = data.float().to(self.dev)
            # label = label.long().to(self.dev)
            label = label.float().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
                # print("output", output)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:

                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1], nargs='+',
                            help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+',
                            help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
