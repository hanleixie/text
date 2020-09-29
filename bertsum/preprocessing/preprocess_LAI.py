#encoding=utf-8


import argparse
import time

from logging import init_logger
import data_builder


def do_format_to_lines(args):
    print(time.clock())
    data_builder.format_to_lines(args)
    print(time.clock())

def do_format_to_bert(args):
    print(time.clock())
    data_builder.format_to_bert(args)
    print(time.clock())


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='', type=str, help='format_raw, format_to_lines or format_to_bert')
    parser.add_argument("-oracle_mode", default='greedy', type=str, help='how to generate oracle summaries, greedy or combination, combination will generate more accurate oracles but take much longer time.')

    parser.add_argument("-raw_path")
    parser.add_argument("-save_path")

    parser.add_argument("-shard_size", default=16000, type=int)  ###change from 2000 to 16000
    parser.add_argument('-min_nsents', default=3, type=int)
    parser.add_argument('-max_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens', default=5, type=int)
    parser.add_argument('-max_src_ntokens', default=200, type=int)

    parser.add_argument('-log_file', default='../logs/LCSTS.log')

    parser.add_argument('-dataset', default='', help='train, valid or test, defaul will process all datasets')

    parser.add_argument('-n_cpus', default=2, type=int)


    args = parser.parse_args()
    init_logger(args.log_file)
    eval('data_builder.'+args.mode + '(args)')
#Step 2 将原始文件转换成json文件存储
#python preprocess_LAI.py -mode format_raw -raw_path ../raw_data -save_path ../raw_data -log_file ../logs/preprocess.log
#Step 3 分句分词 & 分割文件 & 进一步简化格式
#分句分词：首先按照符号['。', '！', '？']分句，若得到的句数少于2句，则用['，', '；']进一步分句
#分割文件：训练集文件太大，分割成小文件便于后期训练。分割后，每个文件包含不多于16000条记录
#BertSum-master_Chinese/src目录下，运行：
#python preprocess_LAI.py -mode format_to_lines -raw_path ../raw_data -save_path ../json_data/LCSTS -log_file ../logs/preprocess.log
#Step 4 句子标注 & 训练前预处理
#句子预处理：找出与参考摘要最接近的n句话(相似程度以ROUGE衡量)，标注为1(属于摘要)
#python preprocess_LAI.py -mode format_to_bert -raw_path ../json_data -save_path ../bert_data -oracle_mode greedy -n_cpus 2 -log_file ../logs/preprocess.log




