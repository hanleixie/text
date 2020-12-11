# -*- coding:utf-8 -*-
# @Time: 2020/12/8 19:57
# @File: compute_metrics.py
# @Software: PyCharm
# @Author: xiehl
# -------------------------
# coding=utf-8

import logging

logger = logging.getLogger(__name__)

from sklearn.metrics import f1_score, accuracy_score

def acc_and_f1(preds, labels):
    acc = accuracy_score(labels, preds)
    # acc = (preds == labels).mean()
    f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def metrics(task_name, preds, labels):
    assert len(preds) == len(labels)

    if task_name:
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)
