# 生成式文本开发流程

## 一、任务描述

### 生成式文本：

给定一段短文本，对其进行摘要总结。例如下面一段话：

-[x] src_text_abs：“彩排、修正、磨合，万众期待下，春晚的演职人员等都在做着最后的冲刺。春晚的节目和演员筛选有哪些标准？羊年春晚将如何回归联欢？又有哪些创新和亮点？本报记者独家专访了羊年春晚总导演哈文。”

-[x] tgt_abs："春晚以观众之心为心"，期望的摘要总结

### 抽取式文本：

给定一段短文本，通过算法选择出其中某一句作为其摘要。例如下面一段话：

-[x] src_text_ext：“彩排、修正、磨合，万众期待下，春晚的演职人员等都在做着最后的冲刺。春晚的节目和演员筛选有哪些标准？羊年春晚将如何回归联欢？又有哪些创新和亮点？本报记者独家专访了羊年春晚总导演哈文。”

-[x] tgt_ext："彩排、修正、磨合，万众期待下，春晚的演职人员等都在做着最后的冲刺"

#### 本次任务为生成式文本摘要生成算法，对抽取式文本摘要算法不做过多阐述。可以通过调整参数进行生成式和抽取式进行转换。

## 二、数据来源

数据来源于新浪微博的大规模中文短文本摘要数据集，数据集中包含了200万真实的中文短文本数据和文本摘要。

数据集一共包含三部分：

-[x] 1、第一部分数据集的主要部分，包含了2400591对（短文本，摘要），这部分数据用来训练生成摘要的模型。

-[x] 2、第二部分包括了10666对人工标注的（短文本，摘要）。

-[x] 3、第三部分包括了1106对，数据独立于第一部分和第二部分。

* 链接：http://icrc.hitsz.edu.cn/Article/show/139.html

## 三、算法选择和使用

### 3.1 算法选择

通过观察数据集的摘要可知，摘要不是简单的从对应文本中抽取某一句话，而是对文本的总结，故抽取式的算法不适合本任务。在选择摘要生成式文本算法时，bertsumabs完美胜任此任务。Bertsumabs是以bert为基础构建的生成式文本摘要算法，其包括encoder层（bert）、decoder层（transformer）、generator层（linear）。

#### encoder层

encoder层由bert模型构成，其输入为input_data（将文本的每个字输入到模型），包含三个向量：（1）input_ids：src （2）token_type_ids：segs （3）attention_mask。，在文本的每句话前面加[CLS]和每句话结束加[SEP]，输出有两种情况：

-[x] 1、`output_layer = model.get_sequence_output()`，获取句子中每一个单词的向量表示，输出shape是[batch.size, seq.length, hidden.size]，这里也包括[CLS]。本次选用的输出方式。

-[x] 2、`output_layer = model.get_pooled_output()`，这个输出是获取句子的output，既上述[CLS]的表示，输出shape是[batch.size, hidden.size]。

* 目的：通过bert预训练模型，可以得到文本语境下的字向量或者句子向量。

#### decoder层

decoder层由transformer模型构成，Transformer作为seq2seq，由经典的Encoder-Decoder模型组成。Encoder层由6个block组成，Decoder层也由6个block组成，Decoder输出的结果经过一个线性层变换后，经过softmax层计算，输出最终的预测结果。

* 目的：通过decoder生成其文本的语义信息。

#### generator层

generator层是由一个线性层经logsoftmax处理，经过LayerNorm后输出，输出维度为[batch.size, vocab.size]。

* 目的：生成一个dim=0时大小为batch.size，dim=1时大小为vocab.size的向量。

#### 推理层

推理层是对generator层的结果进行推理计算，生成合适的文本摘要。

generator层生成的向量在dim=0的维度上，选择最大的分数的那个和所在的位置号，通过预训练模型中的查找所对应的中文字，在max_length的长度内进行文本生成。

* 链接：https://github.com/alebryvas/berk266

* 文章：KDD Converse 2020 paper: http://arxiv.org/abs/2008.09676

### 3.2  数据处理

在使用时使用自己的语料进行训练，首先将中文语料处理成encoder层可识别的数据格式，通过调用以下命令：

```python
python preprocess_LAI.py -mode format_raw -raw_path ../raw_data -save_path ../raw_data -log_file ../logs/preprocess.log
```
raw_path：原始中文语料所在的目录

```python
python preprocess_LAI.py -mode format_to_lines -raw_path ../raw_data -save_path ../json_data/LCSTS -log_file ../logs/preprocess.log
```
save_path：json格式文件保存的文件目录

```python
python preprocess_LAI.py -mode format_to_bert -raw_path ../json_data -save_path ../bert_data -oracle_mode greedy -n_cpus 2 -log_file ../logs/preprocess.log
```
save_path：最终输入到模型的数据文件目录

Bert_data文件包含以下部分：

-[x] src：文本中每个字所对应的位置号码

-[x] tgt：摘要每个字所对应的位置号码

-[x] labels：对应每句话，由0和1组成

-[x] segs：位置编码，0代表奇数、1代表偶数

-[x] clss：文本中每句话开始的位置号码

-[x] src_txt：文本内容

-[x] tgt_txt：摘要内容

### 3.3 模型训练

对处理好的中文语料进行训练，执行如下命令：

```python
python train.py -task abs -mode train -bert_data_path BERT_DATA_PATH -dec_dropout 0.2  -model_path MODEL_PATH -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps NUM -batch_size 140 -train_steps NUMs -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus -1  -log_file LOG_FILE_PATH
```

-[x] -task abs：生成式模型

-[x] -model：训练模型

-[x] -bert_data_path：模型输入数据

-[x] -train_steps：要训练多少次

-[x] -save_checkpoint：对训练的模型多少轮次保存，训练的模型形式为：`model_step_NUM.pt`

### 3.4 模型测试

对训练好的模型进行test测试，执行如下命令：

```python
python train.py -task abs -mode test -batch_size 3000 -test_batch_size 500 -bert_data_path BERT_DATA_PATH -log_file LOG_FILE_PATH -model_path MODEL_PATH -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path RESHLT_PATH -test_from TEST_FROM_PATH
```

-[x] -test_from：训练模型存放目录，如`model_step_NUM.pt`

-[x] -visible_gpus：根据情况选用合适的gpu or cpu

测试完成后会将测试数据集的文本摘要保存到RESULT_PATN中，其中 `xxx.NUM.candidate`

### 3.5 单文本测试

```text
doc：'小张在南京一整形诊所接受鼻部整形，主刀的韩国医生朴光哲，号称“世界鼻部整形泰斗”、“韩国美鼻教父”。没多久，小张的鼻子歪了，假体快滑出鼻尖。记者得知，朴光哲未注册就在中国进行整容手术，属非法行医。整容有风险，提醒爱美的TA！'
predict：'美鼻教父韩国整形泰斗被指非法行医属非法！'
```

单文本测试时，输入为doc，输出为predict。通过调用下面命令：

```python
python model_builder_one.py
```
## 四、项目架构和服务接口调用

### 4.1 项目架构

下图为主要的项目架构：

```bash
.
└── code
    └── bertsumabs
        ├── bert-base-chinese				#中文预训练模型所需文件
        ├── bert_data						#模型输入数据
        ├── json_data						#原始文本处理后保存为json格式
        ├── logs							#日志信息
        ├── models							#保存模型目录
        ├── raw_data						#原始文本目录
        ├── README.md
        ├── requirements.txt
        ├── results							#摘要结果
        ├── src								#主要功能实现目录
        │   ├── model_builder_one.py
        │   ├── models						#算法模型目录
        │   ├── one_predict					#单文本测试模型目录
        │   ├── others						#辅助函数目录
        │   ├── prepro
        │   ├── preprocess.py
        │   ├── train_abstractive.py
        │   ├── train.py
        │   └── translate					#文本摘要生成模型目录
        ├── temp
```
### 4.2 服务接口调用

本项目用 flask 简单封装，目前只支持单文本调用，post 调用，具体请求接口 url、请求参数以及返回体的格式、内容等如下所示：

* **url:**  http://0.0.0.0:8000/bertsumabs

* **params: **

```json
     {
       "doc": "人们通常被社会赋予的'成功'所定义，“做什么工作”“赚多少钱”都用来评判一个人的全部价值，很多人出现身份焦虑。身份焦虑不仅影响幸福感，还会导致精神压力，甚至自杀。如果你也有身份焦虑，这个短片或许会有帮助。"
     }
```
* **response：**

```json
    {
        "result"："人身份焦虑：如何炼成功？"
        "status": "success"
    }
```

## 五、代码解读和问题处理

### 5.1、代码解读

### 5.2、问题处理


## 附：

### 1、bert模型

### 2、transformer模型