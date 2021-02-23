# -*- coding:utf-8 -*-
# @Time: 2021/2/22 18:52
# @File: test.py.py
# @Software: PyCharm
# @Author: xiehl
# -------------------------
import torch
from load_data.utils import *
from logs.log import *
from sql_net.sqlnet import SQLNet
import argparse
from config import *

init_logger()



def epoch_acc(model, batch_size, sql_data, table_data, db_path):
    engine = DBEngine(db_path)
    model.eval()
    perm = list(range(len(sql_data)))
    query, result_set = [], []
    # for st in tqdm(range(len(sql_data)//batch_size+1)):
    # ed = (st+1)*batch_size if (st+1)*batch_size < len(perm) else len(perm)
    # st = st * batch_size

    raw_q_seq = []#问题
    table_ids = []#表id
    q_seq = []
    col_seq = []
    col_num = []
    for i in tqdm(range(0, 1)):
        sql = sql_data[perm[i]]
        raw_q_seq.append(sql['question'])
        q_seq.append([char for char in sql['question']])
        table_ids.append(sql['table_id'])
        col_seq.append([[char for char in header] for header in table_data[sql['table_id']]['header']])
        col_num.append(len(table_data[sql['table_id']]['header']))

    score = model.forward(q_seq, col_seq, col_num)
    pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq)

    for sql_pred, tid in zip(pred_queries, table_ids):

        result = engine.execute(tid, sql_pred['sel'], sql_pred['agg'], sql_pred['conds'], sql_pred['cond_conn_op'], if_test=True)
        if len(result) != 2:
            query_, result_set_ = result, result
        else:
            query_, result_set_ = result
        # print(query_)
        query.append([query_])
        # result_set.append([result_set_])
    return query, result_set

def main(test_sql_data=None, test_table_id=None):
    # 使用列注意力
    args.ca = True
    # 使用保存的模型
    args.restore = True


    if args.toy:
        use_small = True
        gpu = args.gpu
        batch_size = 16
    else:
        use_small = False
        gpu = args.gpu
        batch_size = args.bs

    if test_sql_data and test_table_id:
        test_sql = [{'table_id': test_table_id, 'question': test_sql_data}]
        batch_size = 1
        _, test_table, test_db = load_dataset(use_small=use_small, mode='test')
    else:
        # 加载数据
        test_sql, test_table, test_db = load_dataset(use_small=use_small, mode='test')

    # 加载词向量 维度大小21557
    word_emb = load_word_emb('data/char_embedding.json')
    # 进入sqlnet函数
    model = SQLNet(word_emb, N_word=n_word, use_ca=args.ca, gpu=gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    # 加载保存好的模型
    if args.restore:

        logger.info("从 %s 加载训练好的模型" % model_path)
        model.load_state_dict(torch.load(model_path))
    # 预测
    query, result_set = epoch_acc(model, batch_size, test_sql, test_table, test_db)

    result = {}
    result["query"] = query
    result["result_set"] = result_set

    return result


if __name__ == '__main__':

    table_id = "69d4941c334311e9aefd542696d6e445"
    question = "PE2011大于11或者EPS2011大于11的公司有哪些"

    predict_result = main(test_sql_data=question, test_table_id=table_id)
    print("predict_query:", predict_result[0])

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--bs', type=int, default=16, help='Batch size')
    # # parser.add_argument('--epoch', type=int, default=2, help='Epoch number')
    # parser.add_argument('--gpu', action='store_true', help='训练是否使用gpu')
    # parser.add_argument('--toy', action='store_true', help='If set, use small data for fast debugging')
    # parser.add_argument('--ca', action='store_true', help='是否使用列注意')
    # parser.add_argument('--train_emb', action='store_true', help='Train word embedding for SQLNet')
    # parser.add_argument('--restore', action='store_true', help='Whether restore trained model')
    # parser.add_argument('--logdir', type=str, default='', help='Path of save experiment logs')
    # args = parser.parse_args()
    #
    # # 使用列注意力
    # args.ca = True
    # # 使用保存的模型
    # args.restore = True
    #
    # n_word = 300
    # if args.toy:
    #     use_small = True
    #     gpu = args.gpu
    #     batch_size = 16
    # else:
    #     use_small = False
    #     gpu = args.gpu
    #     batch_size = args.bs
    # learning_rate = 1e-3
    #
    # # 加载数据
    # test_sql, test_table, test_db = load_dataset(use_small=use_small, mode='test')
    # # 加载词向量 维度大小21557
    # word_emb = load_word_emb('data/char_embedding.json')
    # # 进入sqlnet函数
    # model = SQLNet(word_emb, N_word=n_word, use_ca=args.ca, gpu=gpu)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    # # 加载保存好的模型
    # if args.restore:
    #     model_path = 'saved_model/best_model.pt'
    #     logger.info("从 %s 加载训练好的模型" % model_path)
    #     model.load_state_dict(torch.load(model_path))
    #
    # # 用于记录每个子任务的最佳得分,8个子任务
    # best_sn, best_sc, best_sa, best_wn, best_wc, best_wo, best_wv, best_wr = 0, 0, 0, 0, 0, 0, 0, 0
    # best_sn_idx, best_sc_idx, best_sa_idx, best_wn_idx, best_wc_idx, best_wo_idx, best_wv_idx, best_wr_idx = 0, 0, 0, 0, 0, 0, 0, 0
    # best_lf, best_lf_idx = 0.0, 0
    # best_ex, best_ex_idx = 0.0, 0
    #
    # # 预测
    # query, result_set = epoch_acc(model, batch_size, test_sql, test_table, test_db)
