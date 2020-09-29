import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("-task", default='abs', type=str, choices=['ext', 'abs'])
parser.add_argument('-test_from', default='../models/model_step_6500.pt')
parser.add_argument('-visible_gpus', default='-1', type=str)
parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("-max_pos", default=512, type=int)
parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("-dec_hidden_size", default=768, type=int)
parser.add_argument("-dec_heads", default=8, type=int)
parser.add_argument("-dec_ff_size", default=2048, type=int)
parser.add_argument("-dec_dropout", default=0.2, type=float)
parser.add_argument("-enc_hidden_size", default=512, type=int)
parser.add_argument("-enc_ff_size", default=512, type=int)
parser.add_argument("-enc_dropout", default=0.2, type=float)
parser.add_argument("-enc_layers", default=6, type=int)
parser.add_argument("-temp_dir", default='../temp')  # '../temp')
parser.add_argument("-alpha", default=0.6, type=float)
parser.add_argument("-beam_size", default=5, type=int)
parser.add_argument("-min_length", default=15, type=int)
parser.add_argument("-max_length", default=150, type=int)
parser.add_argument("-max_tgt_len", default=140, type=int)
parser.add_argument("-model_path", default='../models/')
parser.add_argument("-result_path", default='../results/pred_one')
parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)
args = parser.parse_args()

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']

vocab_path = '../bert-base-chinese'

server_port = '8000'