# -*- coding:utf-8 -*-
# @Time: 2020/12/8 20:01
# @File: config.py
# @Software: PyCharm
# @Author: xiehl
# -------------------------
# -*- coding:utf-8 -*-
# @Time: 2020/12/8 9:41
# @File: config.py
# @Software: PyCharm
# @Author: xiehl
# -------------------------
import argparse

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--data_dir", default='dataset/THUCNews/data', type=str, required=False,
                    help="输入数据文件地址")
parser.add_argument("--model_type", default='albert', type=str, required=False,
                    help="模型种类")
parser.add_argument("--model_name_or_path", default='prev_trained_model/albert_chinese_small', type=str,
                    required=False,
                    help="模型参数文件地址")
parser.add_argument("--task_name", default='news', type=str, required=False,
                    help="那个种类数据")
parser.add_argument("--output_dir", default='outputs/news', type=str, required=False,
                    help="输出文件地址")
parser.add_argument("--vocab_file", default='prev_trained_model/albert_chinese_small/vocab.txt', type=str)

## Other parameters
parser.add_argument("--config_name", default="", type=str,
                    help="配置文件地址")
parser.add_argument("--tokenizer_name", default="", type=str,
                    help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--cache_dir", default="", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--max_seq_length", default=512, type=int,
                    help="句子最大长度")
parser.add_argument("--do_train", action='store_true',
                    help="训练")
parser.add_argument("--do_eval", action='store_true',
                    help="验证")
parser.add_argument("--do_predict", action='store_true',
                    help="预测")
parser.add_argument("--do_lower_case", action='store_true',
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                    help="批量大小")
parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                    help="验证批量大小")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="Adam学习率")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=3.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")

parser.add_argument('--logging_steps', type=int, default=10,
                    help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=1000,
                    help="每多少部保存一次")
parser.add_argument("--eval_all_checkpoints",type=str,default='do',# action='store_true',
                    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
parser.add_argument("--no_cuda", type=int, default=-1,  # action='store_true',
                    help="GPU")
parser.add_argument('--overwrite_output_dir', action='store_true',
                    help="Overwrite the content of the output directory")
parser.add_argument('--overwrite_cache', action='store_true',
                    help="Overwrite the cached training and evaluation sets")
parser.add_argument('--seed', type=int, default=42,
                    help="随机种子")
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")

args = parser.parse_args()

server_port = 8201


#
tasks_num_labels = {

    'car': ['不好', '好'],
    'news': ['金融', '民生', '房地产', '教育', '科技', '法律', '国际', '运动', '游戏', '娱乐']


}

processors = {

    'car': ['train_1.tsv', 'dev_1.tsv', 'test_1.tsv'],
    'news': ['train_1.txt', 'dev_1.txt', 'test_1.txt']

}

output_modes = {

    'car': "emotion_classification",
    'news': "news_classfification",

}