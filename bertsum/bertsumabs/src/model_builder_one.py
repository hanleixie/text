import copy
import torch
import torch.nn as nn
from pytorch_transformers import BertModel
from models.decoder import TransformerDecoder
from pytorch_transformers import BertTokenizer
from one_predict.predict_one import build_predictor
from one_predict.data_press import BertData
from others.logging import logger, init_logger
from one_predict.config import args, model_flags, vocab_path

def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

class Bert(nn.Module):
    def __init__(self, finetune=False):
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-chinese', cache_dir=args.temp_dir)#r'C:\Users\Administrator\PycharmProjects\one\bertsum-chinese-LAI\temp')
        self.finetune = finetune
    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)#维度(batch, seq_len)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec#(batch_size, sequence_length, hidden_size)


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.finetune_bert)#false, ../temp, ture

        if(args.max_pos>512):#最大不大于512，故此层用不到
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size#此为bert.model中config的vocab_size：21128
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)#同上hidden_size:768# #对摘要进行编码
        if (self.args.share_emb):#False
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        #bertmodel可作为特征提取过程，既此时对应的encoder，transformer作为decoder
        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight
        self.load_state_dict(checkpoint['model'], strict=True)
        self.to(device)

    def forward(self, src, tgt, segs, mask_src):
        top_vec = self.bert(src, segs, mask_src)#src:文本的位置号码， segs：文本的0,1嵌入，奇数为0偶数为1，
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None
'''
def item_to_str(item):
    print('item:', item)
    doc = item.doc

    return doc
'''
def bertsumabs_predict(doc):

    #doc = item_to_str(item)
    data_process = BertData(vocab_path=vocab_path, device='cpu')
    document_splits = data_process.split_long_doc(doc, 521)
    example, doc_sents = data_process.preprocess(document_splits, min_sent_num=2)

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    #test_from = r'C:\Users\Administrator\PycharmProjects\one\yonyou\code\bertsumabs\models\model_step_6500.pt'
    test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    print('Loading checkpoint from %s' % test_from)
    step = int(test_from.split('.')[-2].split('_')[-1])

    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])  # 参数
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])

    model = AbsSummarizer(args, device, checkpoint)
    model.eval()
    #print('Loading checkpoint from %s' % model)
    logger.info('Loading checkpoint from %s' % model)

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab['[unused1]'], 'EOS': tokenizer.vocab['[unused2]'],
               'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused3]']}
    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    pre = predictor.translate(example, step)
    return pre





'''
doc = '人们通常被社会赋予的"成功"所定义，“做什么工作”“赚多少钱”都用来评判一个人的全部价值，很多人出现身份焦虑。身份焦虑不仅影响幸福感，还会导致精神压力，甚至自杀。如果你也有身份焦虑，这个短片或许会有帮助。',
predict = bertsumabs_predict(doc)
print('predict', predict)
'''

























