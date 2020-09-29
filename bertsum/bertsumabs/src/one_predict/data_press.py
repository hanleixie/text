import re
from transformers import BertTokenizer
import torch

def filter(x: str):
    x = str(x).replace('<br>', '。')
    x = filter_chinese_space(x)
    dr = re.compile(r'<[^>]+>', re.S)
    dr2 = re.compile(r'{[^}]+}', re.S)
    if x is None or str(x) == 'Nan' or str(x) == 'nan':
        return x
    x = dr.sub('', x)
    x = dr2.sub('', x)
    x = x.replace('\u3000', '')
    # x = x.replace(' ', '')
    x = x.strip()
    return x

def filter_chinese_space(text: str) -> int:
    '''
    只给中文中的空格去除
    :param x:
    :return:
    '''
    match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
    should_replace_list = match_regex.findall(text)
    order_replace_list = sorted(should_replace_list, key=lambda i: len(i), reverse=True)
    for i in order_replace_list:
        if i == u' ':
            continue
        new_i = i.strip()
        text = text.replace(i, new_i)
    return text

def doc_split(doc: str):
    doc = filter(doc)
    # 给主体文本切成单个句子
    doc_sents = re.split(r"([。|\？|!|；|;])", doc)
    # 过滤空句子
    doc_sents = [str(ds) for ds in doc_sents if ds != '']
    doc_sents.append("")
    doc_sents = ["".join(i) for i in zip(doc_sents[0::2], doc_sents[1::2])]
    doc_sents = [di for di in doc_sents if len(di) >= 2]
    return doc_sents


def split_long_doc(document: str, max_num=510):
    document = filter(document)
    doc_sents = doc_split(document)
    document_list = []
    a_temp_doc = ''
    if len(doc_sents) <= 1:
        return doc_sents

    for si in doc_sents:
        if len(a_temp_doc) + len(si) > max_num:
            document_list.append(a_temp_doc)
            a_temp_doc = si
        else:
            a_temp_doc += si
    if a_temp_doc != '':
        document_list.append(a_temp_doc)
    return document_list

def sent_token_split(doc):
    doc = str(doc)
    doc_split = list(doc)
    return doc_split

class Example(object):
    def __init__(self, data: list, device=None):
        pre_src = [data[0]]
        pre_segs = [data[1]]
        pre_clss = [data[2]]
        src = torch.tensor(pre_src)
        mask_src = ~ (src == 0)

        segs = torch.tensor(pre_segs)
        clss = torch.tensor(pre_clss)
        mask_cls = ~ (clss == -1)

        setattr(self, 'src', src.to(device))
        setattr(self, 'mask_src', mask_src.to(device))
        setattr(self, 'segs', segs.to(device))
        setattr(self, 'clss', clss.to(device))
        setattr(self, 'mask_cls', mask_cls.to(device))

class BertData(object):
    def __init__(self, vocab_path, device='cpu'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=True)
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    def split_long_doc(self, document: str, max_num=510):
        document = filter(document)
        doc_sents = doc_split(document)
        document_list = []
        a_temp_doc = ''
        if len(doc_sents) <= 1:
            return doc_sents

        for si in doc_sents:
            if len(a_temp_doc) + len(si) > max_num:
                document_list.append(a_temp_doc)
                a_temp_doc = si
            else:
                a_temp_doc += si
        if a_temp_doc != '':
            document_list.append(a_temp_doc)
        return document_list

    def preprocess(self, document: str, min_sent_num=1):
        document = filter(document)
        document = re.split(r"([。|\？|!|；|;])", document[2:-2])
        doc_sents = ["".join(i) for i in zip(document[0::2], document[1::2])]
        #doc_sents = doc_split(document)
        if len(doc_sents) <= min_sent_num:
            return None, doc_sents

        #src = [sent_token_split(sent) for sent in doc_sents]
        src_txt = [' '.join(sent) for sent in doc_sents]

        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        # bert accept data type
        src_subtokens = src_subtokens[:510]
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        data = [src_subtoken_idxs, segments_ids, cls_ids]
        example = Example(data, self.device)
        return example, doc_sents
