""" GLUE processors and helpers """

import logging
import os
import torch
from .utils import DataProcessor, InputExample, InputFeatures

logger = logging.getLogger(__name__)

def collate_fn(batch):

    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels

# 对长中文进行处理
import re
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



def convert_examples_to_features(examples, tokenizer,
                                      max_seq_length=512,
                                      label_list=None,
                                      output_mode=None):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):

        if ex_index % 1000 == 0:
            logger.info("Writing example %d" % (ex_index))

        if len(example.text) > max_seq_length:
            example.text = split_long_doc(example.text, max_num=max_seq_length)

        # 将文本变成单个文字
        example.text = doc_split(example.text)
        text_list = []
        for text in example.text:
            text_tokens = tokenizer.tokenize(text)
            text_list.append(text_tokens)

        tokens = []
        token_type_ids = []
        tokens.append("[CLS]")
        token_type_ids.append(0)
        for i in range(len(text_list)):
            tokens.extend(text_list[i])
            tokens.append("[SEP]")
            if i%2 == 0:
                token_type_ids.extend([0] * (len(text_list[i])+1))
            else:
                token_type_ids.extend([1] * (len(text_list[i])+1))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(input_ids)
        input_len = len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)

        # assert len(input_ids) == max_seq_length
        # assert len(attention_mask) == max_seq_length
        # assert len(token_type_ids) == max_seq_length
        if output_mode == "emotion_classification":
            label_id = label_map[example.label]
        else:
            raise KeyError(output_mode)

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label_id,
                          input_len=input_len))
    return features

class CarProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """训练"""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_1.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """验证"""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_1.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """测试"""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_1.tsv")), "test")

    def get_labels(self):
        """获取标签"""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """生成样本数据"""
        examples = []
        for (i, line) in enumerate(lines):
            if i > 0:
                guid = "%s-%s" % (set_type, i-1)
                text = line[2]
                label = line[1]
                examples.append(
                    InputExample(guid=guid, text=text, label=label))
        return examples

tasks_num_labels = {

    'car': 2,

}

processors = {

    'car': CarProcessor,

}

output_modes = {

    'car': "emotion_classification",

}
