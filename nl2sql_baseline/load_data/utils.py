# -*- coding:utf-8 -*-
# @Time: 2021/2/1 10:19
# @File: utils.py
# @Software: PyCharm
# @Author: xiehl
# -------------------------
import numpy as np
from tqdm import tqdm
import json
import records
import sys
sys.path.append('../logs')
from logs.log import *
init_logger()
logger.info('开始数据加载类方法')

agg_dict = {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM"}
cond_op_dict = {0:">", 1:"<", 2:"==", 3:"!="}
rela_dict = {0:'', 1:' AND ', 2:' OR '}

class DBEngine:
    def __init__(self, fdb):
        self.db = records.Database('sqlite:///{}'.format(fdb))#操作sqlite数据库
        self.conn = self.db.get_connection()

    def execute(self, table_id, select_index, aggregation_index, conditions, condition_relation, if_test=False):
        """                     sql_pred['sel'], sql_pred['agg'], sql_pred['conds'], sql_pred['cond_conn_op']
        table_id: id of the queried table.
        select_index: list of selected column index, like [0,1,2]
        aggregation_index: list of aggregation function corresponding to selected column, like [0,0,0], length is equal to select_index
        conditions: [[condition column, condition operator, condition value], ...]
        condition_relation: 0 or 1 or 2
        """
        table_id = 'Table_{}'.format(table_id)

        # 条件数>1 而 条件关系为''
        if condition_relation == 0 and len(conditions) > 1:
            return '预测的条件数和条件关系不相等'
        # 选择列或条件列为0
        if len(select_index) == 0 or len(conditions) == 0 or len(aggregation_index) == 0:
            return '选择列或条件列为0'

        condition_relation = rela_dict[condition_relation]

        select_part = ""
        for sel, agg in zip(select_index, aggregation_index):
            select_str = 'col_{}'.format(sel+1)
            agg_str = agg_dict[agg]
            if agg:
                select_part += '{}({}),'.format(agg_str, select_str)
            else:
                select_part += '({}),'.format(select_str)
        select_part = select_part[:-1]

        where_part = []
        for col_index, op, val in conditions:
            if type(val) == "unicode":
                where_part.append('col_{} {} "{}"'.format(col_index+1, cond_op_dict[op], val.encode('utf-8')))
            else:
                where_part.append('col_{} {} "{}"'.format(col_index+1, cond_op_dict[op], val))
        where_part = 'WHERE ' + condition_relation.join(where_part)

        query = 'SELECT {} FROM {} {}'.format(select_part, table_id, where_part)
        # query = query.decode('utf-8')
        try:
            out = self.conn.query(query).as_dict()
        except:
            return '查询数据库时出错'
        # print(out)
        # result_set = [tuple(sorted(i.values())) for i in out]
        result_set = [tuple(sorted(i.values(), key=lambda x:str(x))) for i in out]

        if if_test:
            return query, result_set

        return result_set

def epoch_train(model, optimizer, batch_size, sql_data, table_data):
    model.train()#继承类的train
    perm = np.random.permutation(len(sql_data))# 生成随机排序len（sql）=41522
    cum_loss = 0.0
    for st in tqdm(range(len(sql_data)//batch_size+1)):
        ed = (st+1)*batch_size if (st+1)*batch_size < len(perm) else len(perm)
        st = st * batch_size
        q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq = to_batch_seq(sql_data, table_data, perm, st, ed)
        # q_seq: sql_data->问题，字符级单位
        # gt_sel_num: number of selected columns and aggregation functions sql_data->每个batch中样本要select的列条件个数：1 or 2
        # col_seq: char-based column name。table_data->每个batch中样本的heads有几个，既有列的名字是什么，字符为单位
        # col_num: number of headers in one table。table_data->中对应sql_data中的table_id的列名有几个，基本上同上的个数
        # ans_seq: (sel, number of conds, sel list in conds, op list in conds)。[聚合函数的个数，要选择的列，具体的聚合函数，conds三原组个数，conds三元组中第一个值（条件列），第二个值（条件类型），sql条件]
        # gt_cond_seq: ground truth of conds。conds三元组
        gt_where_seq = model.generate_gt_where_seq_test(q_seq, gt_cond_seq)#返回conds三元组中的条件值不在问题中出现，标记[0,问题总长度-1]
        gt_sel_seq = [x[1] for x in ans_seq]#选择第几列
        score = model.forward(q_seq, col_seq, col_num, gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq, gt_sel_num=gt_sel_num)
        # sel_num_score, sel_col_score, sel_agg_score, cond_score, cond_rela_score

        # compute loss
        loss = model.loss(score, ans_seq, gt_where_seq)
        cum_loss += loss.data.cpu().numpy()*(ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return cum_loss / len(sql_data)




def to_batch_seq_test(sql_data, table_data, idxes, st, ed):
    q_seq = []
    col_seq = []
    col_num = []
    raw_seq = []
    table_ids = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        q_seq.append([char for char in sql['question']])
        col_seq.append([[char for char in header] for header in table_data[sql['table_id']]['header']])
        col_num.append(len(table_data[sql['table_id']]['header']))
        raw_seq.append(sql['question'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return q_seq, col_seq, col_num, raw_seq, table_ids

def to_batch_seq(sql_data, table_data, idxes, st, ed, ret_vis_data=False):
    q_seq = []# sql_data->问题
    col_seq = []# table_data->每个batch中样本的heads有几个，既有列的名字是什么，字符为单位
    col_num = []# table_data->中对应sql_data中的table_id的列名有几个，基本上同上的个数
    ans_seq = []# [聚合函数的个数，要选择的列，具体的聚合函数，conds三原组个数，conds三元组中第一个值（条件列），第二个值（条件类型），sql条件]
    gt_cond_seq = []# conds三元组
    vis_seq = []# questions和表头的拼接
    sel_num_seq = []#sql_data->每个batch中样本要select的列条件个数：1 or 2
    for i in range(st, ed):
        sql = sql_data[idxes[i]]# select每个batch的元素
        sel_num = len(sql['sql']['sel'])#'sel': 要选择的列
        sel_num_seq.append(sel_num)
        conds_num = len(sql['sql']['conds'])#(条件列，条件类型，条件值)
        q_seq.append([char for char in sql['question']])#问题
        col_seq.append([[char for char in header] for header in table_data[sql['table_id']]['header']])#表头名字
        col_num.append(len(table_data[sql['table_id']]['header']))#有几个表头名字，既有多少列
        ans_seq.append(
            (
            len(sql['sql']['agg']),#选择的列相应的聚合函数
            sql['sql']['sel'],
            sql['sql']['agg'],
            conds_num,
            tuple(x[0] for x in sql['sql']['conds']),
            tuple(x[1] for x in sql['sql']['conds']),
            sql['sql']['cond_conn_op'],
            ))
        gt_cond_seq.append(sql['sql']['conds'])
        vis_seq.append((sql['question'], table_data[sql['table_id']]['header']))#问题和表头的拼接
    if ret_vis_data:
        return q_seq, sel_num_seq, col_seq, col_num, ans_seq, gt_cond_seq, vis_seq
    else:
        return q_seq, sel_num_seq, col_seq, col_num, ans_seq, gt_cond_seq

def to_batch_query(sql_data, idxes, st, ed):
    query_gt = []
    table_ids = []
    for i in range(st, ed):
        sql_data[idxes[i]]['sql']['conds'] = sql_data[idxes[i]]['sql']['conds']
        query_gt.append(sql_data[idxes[i]]['sql'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids




def epoch_acc(model, batch_size, sql_data, table_data, db_path):
    engine = DBEngine(db_path)
    model.eval()
    perm = list(range(len(sql_data)))
    badcase = 0
    one_acc_num, tot_acc_num, ex_acc_num = 0.0, 0.0, 0.0
    for st in tqdm(range(len(sql_data)//batch_size+1)):
        ed = (st+1)*batch_size if (st+1)*batch_size < len(perm) else len(perm)
        st = st * batch_size
        q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq, raw_data = \
            to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        # q_seq: char-based sequence of question
        # gt_sel_num: number of selected columns and aggregation functions, new added field
        # col_seq: char-based column name
        # col_num: number of headers in one table
        # ans_seq: (sel, number of conds, sel list in conds, op list in conds)
        # gt_cond_seq: ground truth of conditions
        # raw_data: ori question, headers, sql
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)#query_gt：sql_data，table_id：table_id
        # query_gt: ground truth of sql, data['sql'], containing sel, agg, conds:{sel, op, value}
        raw_q_seq = [x[0] for x in raw_data]#问题
        try:
            score = model.forward(q_seq, col_seq, col_num)#问题的字符形式，列的字符形式
            pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq)
            # generate predicted format
            one_err, tot_err = model.check_acc(raw_data, pred_queries, query_gt)
        except:
            badcase += 1
            print('badcase', badcase)
            continue
        one_acc_num += (ed-st-one_err)
        tot_acc_num += (ed-st-tot_err)

        # Execution Accuracy
        for sql_gt, sql_pred, tid in zip(query_gt, pred_queries, table_ids):
            ret_gt = engine.execute(tid, sql_gt['sel'], sql_gt['agg'], sql_gt['conds'], sql_gt['cond_conn_op'])
            try:
                ret_pred = engine.execute(tid, sql_pred['sel'], sql_pred['agg'], sql_pred['conds'], sql_pred['cond_conn_op'])
            except:
                ret_pred = None
            ex_acc_num += (ret_gt == ret_pred)
    return one_acc_num / len(sql_data), tot_acc_num / len(sql_data), ex_acc_num / len(sql_data)
        #  子任务损失，总体损失，sql语句返回值损失
















# 加载数据集
def load_data(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}

    for SQL_PATH in sql_paths:
        with open(SQL_PATH, 'r', encoding='utf-8') as inf:
            for idx, line in enumerate(inf):
                sql = json.loads(line.strip())
                if use_small and idx >= 1000:
                    break
                sql_data.append(sql)
        logger.info("Loaded %d data from %s" % (len(sql_data), SQL_PATH))

    for TABLE_PATH in table_paths:
        with open(TABLE_PATH, 'r', encoding='utf-8') as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab[u'id']] = tab
        logger.info("Loaded %d data from %s" % (len(table_data), TABLE_PATH))

    ret_sql_data = []
    for sql in sql_data:
        if sql[u'table_id'] in table_data:
            ret_sql_data.append(sql)

    return ret_sql_data, table_data

def load_dataset(toy=False, use_small=False, mode='train'):
    logger.info("加载数据集")
    # dev_sql, dev_table = load_data('data/val/val.json', 'data/val/val.tables.json', use_small=use_small)
    # dev_db = 'data/val/val.db'
    if mode == 'train':
        train_sql, train_table = load_data('data/train/train.json', 'data/train/train.tables.json', use_small=use_small)
        train_db = 'data/train/train.db'
        dev_sql, dev_table = load_data('data/val/val.json', 'data/val/val.tables.json', use_small=use_small)
        dev_db = 'data/val/val.db'
        return train_sql, train_table, train_db, dev_sql, dev_table, dev_db
    elif mode == 'test':
        test_sql, test_table = load_data('data/test/test.json', 'data/test/test.tables.json', use_small=use_small)
        test_db = 'data/test/test.db'
        # return dev_sql, dev_table, dev_db, test_sql, test_table, test_db
        return test_sql, test_table, test_db


def load_word_emb(file_name):
    logger.info('从%s下载词向量' % file_name)
    f = open(file_name)
    ret = json.load(f)
    f.close()
    return ret









