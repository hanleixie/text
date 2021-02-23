# NL2SQL

## nl2sql 简介

* 什么是nl2sql：

将人类的自然语言自动转化为相应的SQL语句(Structured Query Language结构化查询语言)，使算法能与数据库直接交互、并返回交互的结果，并服务于**子查询预测、SQLNet的替代方案**的研究

* nl2sql 方法：

![方法](https://img-blog.csdnimg.cn/2019071723490235.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzE2OTQ5NzA3,size_16,color_FFFFFF,t_70)

*[参考地址](https://blog.csdn.net/qq_16949707/article/details/96387107?ops_request_misc=%25257B%252522request%25255Fid%252522%25253A%252522161190460916780255260428%252522%25252C%252522scm%252522%25253A%25252220140713.130102334..%252522%25257D&request_id=161190460916780255260428&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-96387107.pc_search_result_no_baidu_js&utm_term=nl2sql%25E6%2595%25B0%25E6%258D%25AE%25E9%259B%2586)*

对上图可有如下解释：

```sql
select $agg{0:"", 1:“AVG”, 2:“MAX”, 3:“MIN”, 4:“COUNT”, 5:“SUM”}
  $column
where
  $column $op{0:">", 1:"<", 2:"", 3:"!="} conn_sql_dict{0:"", 1:“and”, 2:“or”} $column $op{0:">", 1:"<", 2:"", 3:"!="} 
```

**train.json wenjian**

![样例](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW9LCNQzBD4ZJXevD8hjNMN2ETB60uEyPef4Z3HaicYghAkesiafWibdciaKplxYDdeuCzktN7ycWYmHgA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

"agg"选择的是 agg_sql_dict、"cond_conn_op"选择的是 conn_sql_dict、"conds"中条件类型选择的是 op_sql_dict。

![映射关系](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW9LCNQzBD4ZJXevD8hjNMN2xcaDxmicSXlNEtZvJnIndvNLYNOTBibWee5E7IfHARZiakF4FVbuCxfuQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

* 首先模型需要根据 table_id 检索对应的表格，然后再根据 question 再对应表格中检索信息，并生成对应的 SQL 表达式。



## baseline代码解读

在预测模型的结构层次中，将整体任务分为8个子任务。

### sqlnet.py

* 1、`SQLNet`类继承`nn.Module`类，

```python
self.max_col_num = 45	#最大列数目
self.max_tok_num = 200	#最大句子数目
self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'OR', 				'==', '>', '<', '!=', '<BEG>']	#连接关系
self.COND_OPS = ['>', '<', '==', '!=']	#比较关系
```

* 2、在每个问题前面后后面增加两个位置标志符`<BEG>***<END>`。
* 3、如果条件值在问题中，标记`（TRUE，条件值）`，并且增加列表`[0,在问题中的下标索引，问题长度-1]`；否则为`（FALSE，条件值）`。在问题



### data_processing

在每个问题句子前面加标志符号`<BEG>`和句子后面`<END>`，

**Step 1：**

判断`conds`三元组中的`条件值`是否在问题中。

* 情况1：

  如果`cond[2]=条件值`不在问题中，给标记`False`标记；

* 情况2：

  如果`cond[2]=条件值`在问题中，给标记`True`标记。

**Step 2：**

* 对于step1中情况1，找到`条件值`的位置编码，但最终返回`空列表[]`；
* 对于step1中情况2，因`条件值`不在问题中，返回`[0，Q_len（问题长度）]`。

### model.forward()

**Step 1：**

* 1、对问题进行向量（三维向量）编码，每个问题的长度；
* 2、对列名进行向量化（三维向量）编码，每个列名的长度，每个表有多少个列；

**Step 2：**

* `select_number.py`，**预测列个数**
* **对列进行编码**
  * 1、对列名字进行编码。首先将`[col_num, max_word_num, emb_size_first]`通过**双向lstm**编码为`[col_num, emb_size_second]`，然后将编码后的列向量变为`[batch_size, max_seq_len, emb_size_second]`；
  * 2、通过一个线性变换将编码后的列向量变为`[batch_size, max_seq_len]`；
  * 3、对于因统一最大列个数补全的那些赋予一个很小的负值（-10000000）；
  * 4、当列编码后经过softmax后的每列得分乘以编码后的列向量，并在第二个维度求和得到维度为`[batch_size, emb_size_third]`。每个问题样本所对应的列编码向量。
  * 5、经线性变换得到`[16, 200]`，后经维度变换为`[16, 4, 50]`，再交换维度得`[4, 16, 50]`；
  * **函数中没有返回列编码的内容**
* **对问题编码**
  * 1、对列名字进行编码。首先将`[batch_size, max_word_num, emb_size_first]`通过**双向lstm**编码为`[batch_size, max_word_num, emb_size_second]`，然后将编码后的问题向量经线性变换为`[batch_size, max_seq_len]`；
  * 2、对于因统一最大问题字个数补全的那些赋予一个很小的负值（-10000000）；
  * 3、当问题编码后经过softmax后的每个问题得分乘以编码后的问题向量，并在第二个维度求和得到维度为`[batch_size, emb_size_4th]`。每个问题样本所对应的问题编码向量。
  * 4、经过线性变换、tanh函数、线性编码后得到`[batch_size, score]=[16, 4]`；
  * 5、返回问题编码，既第四步的输出。

**Step 3：**

* `selection_predict.py`，挑选列的得分。
* 1、对问题和列进行编码，步骤同`step1`，列编码为`[batch_size, max_seq_len, emb_size_second]=[16, 14, 100]`、问题编码`[batch_size, max_word_num, emb_size_second]=[16, 47, 100]`；
* 2、将问题编码向量经线性变换后交换1、2维度得`[16, 100, 47]`，与列编码进行想乘的`[16, 47, 14]`；    **？？意义何在？？**
* 3、对于因统一最大问题字个数补全的那些赋予一个很小的负数（-100）；
* 4、将步骤二的输出经维度变换`view(-1, 47)`后经softmax后再经`.view(16, -1, 47)`，然后问题编码乘以它并在最后维度求和，得维度`[16, 14, 100]`。
* 5、步骤四输出和列编码均经过一次线性变换后相加，再经过一次线性变化输出为`[16, 14]`，对于因统一最大问题字个数补全的那些赋予一个很小的负数（-100）；
* 步骤五的结果作为输出。步骤四中的求和是因为所挑选的列个数和问题和列有关。

**Step 4：**

`aggregator_predict.py`，聚合函数（6个）的预测：

* 1、对问题和列进行编码，步骤同`step1`，列编码为`[batch_size, max_seq_len, emb_size_second]=[16, 14, 100]`、问题编码`[batch_size, max_word_num, emb_size_second]=[16, 47, 100]`；
* 2、根据列编码拿出所选择的某些列编码，得`[16, 4, 100]`，第二维代表最多选择四列，当小于四列时，其余列为**第0列**的编码复制。
* 3、问题经线性变换后和步骤2的输出相乘；
* 4、对于因统一最大问题字个数补全的那些赋予一个很小的负数（-100）；
* 5、步骤2的和步骤4的输出相加后经过线性变换、tanh函数、线性编码后得到`[16, 4，6]`；
* 步骤五的结果作为输出。

**Step 5：**

`where_relation.py`，

* 1、对列进行编码得`[16, 14, 100]`
* 2、步骤1的结果经过一次线性变换得`[16, 14]`；
* 3、对于因统一最大问题字个数补全的那些赋予一个很小的负数（-1000000）；
* 4、步骤3的结果经softmax后增加最后面的维度并和问题编码相乘，再第一维度求和得
* 5、问题编码和上述step相同，最后得`[16, 3]`；

**Step 6：**

`sqlnet_condition_predict.py`，

**conds num预测**

* 1、对列进行编码，`[16, 4, 100]`；
* 2、经线性变换得`[16, 14]`；（列注意力机制？）
* 3、对于因统一最大问题字个数补全的那些赋予一个很小的负数（-1000000）；
* 4、步骤二的输出和步骤一的输出相乘，经过两次线性变换后作为lstm的隐层状态，再经过一次线性变换得`[16, 47]`，对于因统一最大问题字个数补全的那些赋予一个很小的负数（-1000000）；
* 5、最后的`[16, 5]`

**conds columns预测**

* 1、问题编码和列编码，然后相乘交换一二维度得`[16, 14, 47]`，然后softmax后和问题编码相乘得`[16, 14, 100]`；
* 2、经线性变换得`[16, 14]`

**conditions operator预测**

* 1、对问题和列进行编码，对列编码操作得`[16, 4, 100]`，对于因统一最大问题字个数补全的那些赋予一个很小的负数（-1000000），softmax后增加最后面的维度并和问题编码相乘，再第一维度求和得`[16, 4, 100]`；
* 2、经线性变换得`[16, 4, 4]`

**string预测**

那些不在问题中的conds条件值

* 1、`[16, 4, 47]`









### word_enbedding.py

* 1、`WordEmbedding()`类继承`nn.Module`类；
* 2、根据`trainable=**`选择是否要自己训练词向量。
* 3、`nn.LSTM`采用输入维度`input_size=300,hidden_size=50,num_layers=2,bidiredtional=True`的参数结构，每个词向量的维度为`300`维。

### 任务代码方面

* 获取每个字的向量表示，包括`END和BEG`，其用全为零的向量表示；
* 最终的为`[batch_size, max_len, embedding_size]`三维向量，如：`[16, 47, 300]`；
* 同时返回每个句子的长度。



### 子任务一、select_number.py

**预测列数目**

* 1、`SelNumPredictor()`类继承`nn.Module`类；
* 2、选用双向长短时网络和线性网络组合。



### selection_predict.py

**预测那个列被选中了**

* 1、`SelPredictor()`类继承`nn.Module`类；
* 2、选用双向长短时网络和线性网络组合。



### aggregator_predict.py

**预测相应选定列的聚合函数**

* 1、`AggPredictor()`类继承`nn.Module`类；
* 2、选用双向长短时网络和线性网络组合。



### sqlnet_condition_predict.py

**预测条件数、条件列、条件操作和条件值(内部比较复杂)**

* 1、`SQLNetCondPredictor()`类继承`nn.Module`类；
* 2、选用双向长短时网络和线性网络组合。



### where_relation.py

**预测条件关系，如“and”、“or”**

* 1、`WhereRelationPredictor()`类继承`nn.Module`类；
* 2、选用双向长短时网络和线性网络组合。