#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import os
from others.logging import init_logger
from train_abstractive import validate_abs, train_abs, baseline, test_abs, test_text_abs
from train_extractive import train_ext, validate_ext, test_ext

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default='abs', type=str, choices=['ext', 'abs'])
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-mode", default='test', type=str, choices=['train', 'validate', 'test', 'test_text'])
    parser.add_argument("-bert_data_path", default=r'C:\Users\Administrator\PycharmProjects\one\bertsum-chinese-LAI\bert_data\LCSTS')#'../bert_data_new/cnndm')
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-result_path", default='../results/LCSTS')
    parser.add_argument("-temp_dir", default=r'C:\Users\Administrator\PycharmProjects\one\bertsum-chinese-LAI\temp')#'../temp')

    parser.add_argument("-batch_size", default=140, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)

    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    # params for EXT
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=1, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)



    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)


    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='../logs/LCSTS.log')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default='../models/model_step_6500.pt')
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)#1
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    init_logger(args.log_file)
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1#-1

    if (args.task == 'abs'):
        if (args.mode == 'train'):
            train_abs(args, device_id)
        elif (args.mode == 'validate'):
            validate_abs(args, device_id)
        elif (args.mode == 'lead'):
            baseline(args, cal_lead=True)
        elif (args.mode == 'oracle'):
            baseline(args, cal_oracle=True)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_abs(args, device_id, cp, step)
        elif (args.mode == 'test_text'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_text_abs(args, device_id, cp, step)

    elif (args.task == 'ext'):
        if (args.mode == 'train'):
            train_ext(args, device_id)
        elif (args.mode == 'validate'):
            validate_ext(args, device_id)
        if (args.mode == 'test'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_ext(args, device_id, cp, step)
        elif (args.mode == 'test_text'):
            cp = args.test_from
            try:
                step = int(cp.split('.')[-2].split('_')[-1])
            except:
                step = 0
            test_text_abs(args, device_id, cp, step)



#python train.py -task abs -mode train -bert_data_path ../../bert_data/how2 -ext_dropout 0.1 -model_path ../models -lr 2e-3 -visible_gpus -1 -report_every 50 -save_checkpoint_steps 10 -batch_size 3000 -train_steps 50 -accum_count 2 -log_file ../logs/ext_bert_cnndm -use_interval true -warmup_steps 10000 -max_pos 512

# python train.py -task abs -mode test -batch_size 3000 -test_batch_size 500 -bert_data_path ../../bert_data/how2 -log_file ../logs/val_abs_bert_cnndm -model_path ../models -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../logs/abs_bert_how2 -test_from ../models/model_step_100.pt
# python train.py -task abs -mode validate -batch_size 3000 -test_batch_size 500 -bert_data_path BERT_DATA_PATH -log_file ../logs/val_abs_bert_cnndm -model_path MODEL_PATH -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -min_length 20 -max_length 100 -alpha 0.9 -result_path ../logs/abs_bert_cnndm



# python train.py -task abs -mode train -bert_data_path /home/grizzly/xiehanlei/berk266/data/inputs/how2 -ext_dropout 0.1 -model_path ../models -lr 2e-3 -visible_gpus -1 -report_every 50 -save_checkpoint_steps 10 -batch_size 3000 -train_steps 50 -accum_count 2 -log_file ../logs/abs_bert_how2 -use_interval true -warmup_steps 10000 -max_pos 512
#python train.py -task ext -mode test_text -batch_size 3000 -test_batch_size 500 -bert _data_path ../../bert_data/how2 -log_file ../logs/val_abs_bert_cnndm -model_path ../models -sep_optim true -use_interval true -visible_gpus -1 -max_pos 512 -min_leng th 20 -max_length 100 -alpha 0.9 -result_path ../logs/val_abs_bert_cnndm -test_from ../models/model_step_50.pt


#python train.py -mode train -accum_count 5 -batch_size 300 -dec_dropout 0.1 -log_file ../logs/LCSTS_baseline -lr 0.05 -model_path ../models -save_checkpoint_steps 200 -seed 777 -sep_optim false -train_steps 1000 -use_bert_emb true -use_interval true -warmup_steps 8000  -visible_gpus -1 -max_pos 512 -report_every 50 -enc_hidden_size 512  -enc_layers 6 -enc_ff_size 2048 -enc_dropout 0.1 -dec_layers 6 -dec_hidden_size 512 -dec_ff_size 2048 -task abs

# python train.py -task abs -mode test -batch_size 3000 -test_batch_size 500 -log_file ../logs/test_abs_basline_LCSTS -model_path ../models -sep_optim true -use_interval true -visible_gpus -1 -max_pos 512 -min_length 20 -max_length 100 -alpha 0.9 -result_path ../results/abs_bert_LCSTS
#bert_abs
#python train.py -task abs -mode train -dec_dropout 0.2  -model_path ../models -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 500 -batch_size 140 -train_steps 20000 -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus -1  -log_file ../logs/abs_bert_LSTCS
