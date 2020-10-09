# 生成式文本开发流程

## 一、任务描述

### 生成式文本：

给定一段短文本，对其进行摘要总结。例如下面一段话：
* src_text_abs：“彩排、修正、磨合，万众期待下，春晚的演职人员等都在做着最后的冲刺。春晚的节目和演员筛选有哪些标准？羊年春晚将如何回归联欢？又有哪些创新和亮点？本报记者独家专访了羊年春晚总导演哈文。”

* tgt_abs："春晚以观众之心为心"，期望的摘要总结

### 抽取式文本：

给定一段短文本，通过算法选择出其中某一句作为其摘要。例如下面一段话：

* src_text_ext：“彩排、修正、磨合，万众期待下，春晚的演职人员等都在做着最后的冲刺。春晚的节目和演员筛选有哪些标准？羊年春晚将如何回归联欢？又有哪些创新和亮点？本报记者独家专访了羊年春晚总导演哈文。”

* tgt_ext："彩排、修正、磨合，万众期待下，春晚的演职人员等都在做着最后的冲刺"

#### 本次任务为生成式文本摘要生成算法，对抽取式文本摘要算法不做过多阐述。可以通过调整参数进行生成式和抽取式进行转换。

## 二、数据来源

数据来源于新浪微博的大规模中文短文本摘要数据集，数据集中包含了200万真实的中文短文本数据和文本摘要。

数据集一共包含三部分：

* 1、第一部分数据集的主要部分，包含了2400591对（短文本，摘要），这部分数据用来训练生成摘要的模型。

* 2、第二部分包括了10666对人工标注的（短文本，摘要）。

* 3、第三部分包括了1106对，数据独立于第一部分和第二部分。

* 链接：http://icrc.hitsz.edu.cn/Article/show/139.html

## 三、算法选择和使用

### 3.1 算法选择

通过观察数据集的摘要可知，摘要不是简单的从对应文本中抽取某一句话，而是对文本的总结，故抽取式的算法不适合本任务。在选择摘要生成式文本算法时，bertsumabs完美胜任此任务。Bertsumabs是以bert为基础构建的生成式文本摘要算法，其包括encoder层（bert）、decoder层（transformer）、generator层（linear）。

#### encoder层

encoder层由bert模型构成，其输入为input_data（将文本的每个字输入到模型），包含三个向量：（1）input_ids：src （2）token_type_ids：segs （3）attention_mask。，在文本的每句话前面加[CLS]和每句话结束加[SEP]，输出有两种情况：

* 1、`output_layer = model.get_sequence_output()`，获取句子中每一个单词的向量表示，输出shape是[batch.size, seq.length, hidden.size]，这里也包括[CLS]。本次选用的输出方式。

* 2、`output_layer = model.get_pooled_output()`，这个输出是获取句子的output，既上述[CLS]的表示，输出shape是[batch.size, hidden.size]。

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
python preprocess_LAI.py -mode format_raw -raw_path ../bertsumabs/raw_data -save_path ../bertsumabs/raw_data -log_file ../bertsumabs/logs/preprocess.log
```
raw_path：原始中文语料所在的目录

```python
python preprocess_LAI.py -mode format_to_lines -raw_path ../bertsumabs/raw_data -save_path ../bertsumabs/json_data/LCSTS -log_file ../bertsumabs/logs/preprocess.log
```
save_path：json格式文件保存的文件目录

```python
python preprocess_LAI.py -mode format_to_bert -raw_path ../bertsumabs/json_data -save_path ../bertsumabs/bert_data -oracle_mode greedy -n_cpus 2 -log_file ../bertsumabs/logs/preprocess.log
```
save_path：最终输入到模型的数据文件目录

Bert_data文件包含以下部分：

* src：文本中每个字所对应的位置号码

* tgt：摘要每个字所对应的位置号码

* labels：对应每句话，由0和1组成

* segs：位置编码，0代表奇数、1代表偶数

* clss：文本中每句话开始的位置号码

* src_txt：文本内容

* tgt_txt：摘要内容

数据处理的在**preprocessing**目录下

### 3.3 模型训练

对处理好的中文语料进行训练，执行如下命令：

```python
python train.py -task abs -mode train -bert_data_path BERT_DATA_PATH -dec_dropout 0.2  -model_path MODEL_PATH -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps NUM -batch_size 140 -train_steps NUMs -report_every 50 -accum_count 5 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus -1  -log_file LOG_FILE_PATH
```

**参数可在配置文件中修改 .. / one_predict / config.py**

* -task abs：生成式模型

* -model：训练模型

* -bert_data_path：模型输入数据

* -train_steps：要训练多少次

* -save_checkpoint：对训练的模型多少轮次保存，训练的模型形式为：`model_step_NUM.pt`

### 3.4 模型测试

对训练好的模型进行test测试，执行如下命令：

```python
python train.py -task abs -mode test -batch_size 3000 -test_batch_size 500 -bert_data_path BERT_DATA_PATH -log_file LOG_FILE_PATH -model_path MODEL_PATH -sep_optim true -use_interval true -visible_gpus 1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path RESHLT_PATH -test_from TEST_FROM_PATH
```

**参数可在配置文件中修改 .. / one_predict / config.py**

* -test_from：训练模型存放目录，如`model_step_NUM.pt`

* -visible_gpus：根据情况选用合适的gpu or cpu

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

## 五、问题处理

在使用bertsumabs前，需要先下载pyrouge，而安装pyrouge之前，应先下载安装Perl。
#### 安装perl
一般 Mac 和 Linux 都有自带的 perl，使用 perl -v检查其版本， 若版本较低可先升级 perl。
#### 安装XML::DOM
使用 cpanm 安装 perl 模块，没有的话先 brew install cpanm / apt install cpanm
接下来就容易多了：
* sudo cpan XML::DOM， 注意使用 sudo 安装。

* sudo cpan XML::Parser

* sudo cpan XML::RegExp

* sudo cpan XML::Parser::PerlSAX
#### 下载和配置 ROUGE 155
* 下载：git clone https://github.com/summanlp/evaluation

* 配置：`export ROUGE_EVAL_HOME="yourPath/evaluation/ROUGE-RELEASE-1.5.5/data/"`，注意路径填写你 clone 下来的文件位置，直至 data 目录。

* 到文件夹下：
```
cd ROUGE-RELEASE-1.5.5
vim ROUGE-1.5.5.pl，查看第一行 perl 路径是否正确。
默认为 #!/usr/bin/perl -w，使用which perl检查结果是否一致，不一致则修改。
```

* 执行测试文件：`perl runROUGE-test.pl`，不报错则安装成功！

**注：**此为在Linux和Ubuntu下安装使用，Windows暂不考虑！

## 附：
### 1、bert模型
BERT (Bidirectional Encoder Representations from Transformers)由Google AI Language 发布了论文 BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding。

bert采用双向的transformer用于语言模型，它对语境的理解会比向的语言模型更深刻。但bert只使用了transformer的encoder部分，且由多个transformer encoder堆叠组成，其输出维度和输入维度相同。如图所示：

![bert](https://www.lyrn.ai/wp-content/uploads/2018/11/transformer-1024x495.png)

*图片 by Rani Horev*



### 2、transformer模型
Transformer模型来自论文Attention Is All You Need，这个模型最初是为了提高机器翻译的效率，它的Self-Attention机制和Position Encoding可以替代RNN。

transformer分成Encoder和Decoder两个部分，Encoder由6个结构一样的encoder堆叠而成，Decoder同理，如图所示：

![stacked Encoder and Decoder](http://fancyerii.github.io/img/transformer/The_transformer_encoder_decoder_stack.png)

更详细信息如图所示：

![transformer](http://fancyerii.github.io/img/transformer/transformer_resideual_layer_norm_3.png)

参考：http://fancyerii.github.io/2019/03/09/transformer-illustrated/










