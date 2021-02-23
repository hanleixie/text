# NL2SQL

## 一、nl2sql 简介

* 什么是nl2sql：

将人类的自然语言自动转化为相应的SQL语句(Structured Query Language结构化查询语言)，使算法能与数据库直接交互、并返回交互的结果。

* 目的：

nl2sql的目的是让不懂数据库编写操作的人员通过输入要查询的自然语言，返回想要的查询结果。例如：

```txt
input:二零一九年第四周大黄蜂和密室逃生这两部影片的票房总占比是多少呀
output:30%
```

nl2sql 的实现方法：

![方法](https://img-blog.csdnimg.cn/2019071723490235.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzE2OTQ5NzA3,size_16,color_FFFFFF,t_70)

*[参考地址](https://blog.csdn.net/qq_16949707/article/details/96387107?ops_request_misc=%25257B%252522request%25255Fid%252522%25253A%252522161190460916780255260428%252522%25252C%252522scm%252522%25253A%25252220140713.130102334..%252522%25257D&request_id=161190460916780255260428&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-96387107.pc_search_result_no_baidu_js&utm_term=nl2sql%25E6%2595%25B0%25E6%258D%25AE%25E9%259B%2586)*

对上图可有如下解释：

```sql
select $agg{0:"", 1:“AVG”, 2:“MAX”, 3:“MIN”, 4:“COUNT”, 5:“SUM”}
  $column
where
  $column $op{0:">", 1:"<", 2:"", 3:"!="} conn_sql_dict{0:"", 1:“and”, 2:“or”} $column $op{0:">", 1:"<", 2:"", 3:"!="} 
```

### 1.1 现有技术总结

**1、Seq2SQL模型**

包含两个部分：

1. **Augmented pointer network**

特点是输出序列中的token是从输入序列token中选取出来的。输入序列是将表格的列名**、**SQL词典、问题拼接起来形成的。其中SQL的词典包括select、where、count、min、max等固定词汇。

2. **Seq2SQL**

模型包括3个组件：

（1）聚合分类器：选择聚合类型，比如sum、count等，也可以没有为null；

（2）select列的pointer：采用pointer network选择要查询的列；

（3）where条件解码器：采用pointer network选择语句。

[论文地址](https://arxiv.org/abs/1711.04436)

**2、SQLNet**

SQLNet的基本解决思想：

- 采用插槽填充的思想，通过对自然语言问句的刨析来向SQL结构中的插槽填充各种值。这样有效地避免了seq2seq中where语句的顺序问题。
- 引入了两种方法
  - Sequence-to-set(序列到集合)
  - Column attention(列注意力)

[sqlnet论文](https://arxiv.org/pdf/1711.04436.pdf)

**3、Bidirectional Attention for SQL Generation**

基于SQL查询的结构，将模型分为三个子模块，并为每个子模块设计特定的深度神经网络。 从类似的机器阅读任务中汲取灵感，采用了**双向注意力机制**和卷积神经网络（CNN）进行**字符级嵌入**来改善结果。 

**4、TypeSQL: Knowledge-based Type-Aware Neural Text-to-SQL Generation**

将问题视为插槽填充任务。 另外，TYPESQL利用类型信息更好地理解自然语言问题中的稀有实体和数字。

![typesql](https://img-blog.csdnimg.cn/img_convert/f246fe9b56a25da613fdcc3f78669874.png)

**5、X-SQL: Reinforce Schema Representation With Context**

将自然语言解析为SQL查询的新网络体系结构。 X-SQL提议通过BERT样式的预训练模型的上下文输出以及类型信息来学习用于下游任务的新模式表示，从而增强结构模式表示的能力。

模型包括三个部分：

1. 序列编码器：类似BERT，不同点在于，一是我们给每个表增加了一个特殊的空列[EMPTY]，二是segment embedding替换为类型embeding，包括问题、类别列、数字列、特殊空列，共4种类型，三是采用MT-DNN而不是BERT-Large来初始化。

2. 上下文增强的schema编码器：根据表格每列的tokens的编码来得到相应列的表示hCi，利用attention，如图(a)。

3. 输出层：将任务分解为6个子任务，每个均采用更简单的结构，采用LayerNorm，输入为schema的表示hCi和上下文表示hCTX。

![X-sql](https://img-blog.csdnimg.cn/img_convert/517ed8270594eef6dd8c8ac2e4162e38.png)

## 二、NL2SQL数据集TableQA

**数据集为单表解析数据集**，数据集包含有约 4500 张表格，且基于这些表格提出了 50000 条自然语言问句，以及对应的 SQL 语句，和WikiSQL对比：

|                |     WikiSQL      |  TableQA  |
| :------------: | :--------------: | :-------: |
|      语种      |       英文       |   中文    |
|      数量      |       8w+        |    5w     |
|      难易      |       简单       |   较难    |
| Select字段数量 |        1         |   [1,2]   |
| Where条件数量  |     单个为主     | 多个为主  |
| Where条件操作  |       AND        | [AND, OR] |
|  Value标准度   | 存在于数据库表中 | 形式多样  |

数据集主要由 3 个文件组成，例如训练集中包括 train.json、train.tables.json 及 train.db。其中 train.json 储存了所有提问与对应的 SQL 表达式，train.tables.json 则储存了所有表格。另外 train.db 为 SQLite 格式的数据库文件，它是为执行模型生成的 SQL 语句而准备的。

### 2.1 其它数据集

**WikiSQL**

WikiSQL数据集是Salesforce在2017年提出的大型标注NL2SQL数据集，也是目前规模最大的NL2SQL数据集。它包含了 24,241张表，80,645条自然语言问句及相应的SQL语句。目前学术界的预测准确率可达91.8%。WikiSQL的问题长度8-15个词居多，查询长度8-11个词居多，表的列数5-7个居多，另外，大多数问题是what类型，其次是which、name、how many、who等类型。

![WikiSQL](https://img-blog.csdnimg.cn/20191030160826643.jpeg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JsYWNrX3NvaWw=,size_16,color_FFFFFF,t_70)

**Spider**

Spider数据集是耶鲁大学于2018年新提出的一个较大规模的NL2SQL数据集。该数据集包含了10,181条自然语言问句，分布在200个独立数据库中的5,693条SQL，内容覆盖了138个不同的领域。虽然在数据数量上不如WikiSQL，但Spider引入了更多的SQL用法，例如Group By、Order By、Having等高阶操作，甚至需要Join不同表，更贴近真实场景，所以难度也更大。目前准确率最高只有54.7%。

![Spider](https://img-blog.csdnimg.cn/20191030161022508.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JsYWNrX3NvaWw=,size_16,color_FFFFFF,t_70)

**WikiTableQuestions**

该数据集是斯坦福大学于2015年提出的一个针对维基百科中那些半结构化表格问答的数据集，内部包含22,033条真实问句以及2,108张表格。由于数据的来源是维基百科，因此表格中的数据是真实且没有经过归一化的，一个cell内可能包含多个实体或含义，比如「Beijing, China」或「200 km」；同时，为了很好地泛化到其它领域的数据，该数据集测试集中的表格主题和实体之间的关系都是在训练集中没有见到过的。

![WikiTableQuestions](https://img-blog.csdnimg.cn/20191030161128379.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JsYWNrX3NvaWw=,size_16,color_FFFFFF,t_70)

*下载地址*

*[WikiSQL](https://github.com/salesforce/WikiSQL)*

*[Spider](https://yale-lily.github.io/spider)*

*[WikiTableQuestions](https://github.com/ppasupat/WikiTableQuestions)*

*[TableQA](https://tianchi.aliyun.com/competition/entrance/231716/information)*

**train.json 文件**

![样例](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW9LCNQzBD4ZJXevD8hjNMN2ETB60uEyPef4Z3HaicYghAkesiafWibdciaKplxYDdeuCzktN7ycWYmHgA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

`"agg"`选择的是 `agg_sql_dict`、`"cond_conn_op"`选择的是 `conn_sql_dict`、`"conds"`中条件类型选择的是 `op_sql_dict`。

![映射关系](https://mmbiz.qpic.cn/mmbiz_png/KmXPKA19gW9LCNQzBD4ZJXevD8hjNMN2xcaDxmicSXlNEtZvJnIndvNLYNOTBibWee5E7IfHARZiakF4FVbuCxfuQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**模型需要根据"table_id"和"question"预测出完整的 SQL 表达式 **

**具体数据如下：**

```json
train.json:
{"table_id": "4d29d0513aaa11e9b911f40f24344a08", "question": "二零一九年第四周大黄蜂和密室逃生这两部影片的票房总占比是多少呀", "sql": {"agg": [5], "cond_conn_op": 2, "sel": [2], "conds": [[0, 2, "大黄蜂"], [0, 2, "密室逃生"]]}}
train.tables.json:
{"rows": [["死侍2：我爱我家", 10637.3, 25.8, 5.0], ["白蛇：缘起", 10503.8, 25.4, 7.0], ["大黄蜂", 6426.6, 15.6, 6.0], ["密室逃生", 5841.4, 14.2, 6.0], ["“大”人物", 3322.9, 8.1, 5.0], ["家和万事惊", 635.2, 1.5, 25.0], ["钢铁飞龙之奥特曼崛起", 595.5, 1.4, 3.0], ["海王", 500.3, 1.2, 5.0], ["一条狗的回家路", 360.0, 0.9, 4.0], ["掠食城市", 356.6, 0.9, 3.0]], "name": "Table_4d29d0513aaa11e9b911f40f24344a08", "title": "表3：2019年第4周（2019.01.28 - 2019.02.03）全国电影票房TOP10", "header": ["影片名称", "周票房（万）", "票房占比（%）", "场均人次"], "common": "资料来源：艺恩电影智库，光大证券研究所", "id": "4d29d0513aaa11e9b911f40f24344a08", "types": ["text", "real", "real", "real"]}

```

根据上面的例子，期待返回sql语句：

```sql
select sum('票房占比') from 'Table_4d29d0513aaa11e9b911f40f24344a08' where ('影片名称'='大黄蜂' and '影片名称'='密室逃脱' )
```

*具体而言，模型应该通过问题中的「票房总占比是多少」确定需要选择第三个特征列「票房占比（%）」，即"sel": [2]；以及对应的聚合函数「SUM」，即"agg": [5]。通过问题中的「大黄蜂和密室逃生」从影片名称中确定"大黄蜂"和"密室逃生"两个条件，同时这两个条件间的关系应该为「AND」，即确定"conds"和"cond_conn_op"分别是什么。*

**关于conds的value预测问题**

根据统计（train.json)，条件值完全包含于question中的大概有82%，剩下的一部分是在表格中（如果cond_op为“=”理论上应该肯定在表格中），还有一些需要做转换，具体如下几种情况：

1. 需要年份补全，如“有几家传媒公司16年为了...”，cond_value为“2016”；

2. 需要单位转换，如“哪些楼盘平均售价低于2万”，cond_value为“20000”；

3. 需要数值转换，如“价格高于十元或者涨幅大于百分之八”，cond_value为“8”（col的单位为%）；

4. 需要名称消歧，如“建发总市值多少”，cond_value为“建发股份”。

## 三、开发流程

### 3.1 nl2sql在做什么样的任务

- 1 、判定`agg`，选择的聚合函数是什么？
- 2、 判定`column`，又可分为列个数`sel_num`多少个？列值`sel_pred`是什么？
- 3、 判定`cond_pred`，`where`后面的条件语句是什么？
- 4、 判定`where_rela_pred`，`where`后面条件的组合关系是什么，`and或者or`？

**将上述任务合并成一句有效的SQL表达式，整体而言，首先模型需要根据 table_id 检索对应的表格，然后再根据 question 再对应表格中检索信息，并生成对应的 SQL 表达式。**

### 3.2 开发过程可做什么事情？

**3.2.1 数据预处理**

1. 年份补全：“有几家传媒公司16年为了...”，（16，2016）；
2. 单位转换：“哪些楼盘平均售价低于2万”，（2万，20000）；
3. 数值转换，“价格高于十元或者涨幅大于百分之八”，（（十，10），（百分之八，8%））；
4. 名称消歧：“建发总市值多少”，（建发，建发股份）

**3.2.2 模型选择分析**

1. **使用字符级预训练词嵌入**，词向量文件为`char_embedding.json`，其是在10G 大小的百度百科和维基百科语料上采用 Skip-Gram 训练。**以字为单位可以避免专有名词带来的 OOV 问题以及分词错误所带来的错误累积**。
2. 既然以字为单位，则可使用各种基于`bert`的模型。
3. 以字为单位，也可使用基于`lstm`的模型。
4. 条件值完全包含在question中，正常做，**当条件值不在question中，可作为信息抽取来做或者编辑距离做**。

**3.2.3 任务细分**

将nl2sql任务细分为多个子任务：

1. Sel-Num：选择的列个数
2. Sel-Col:：选择那个列
3.  Sel-Agg：选择的聚合函数
4.  W-Num：选择的条件个数
5.  W-Col：选择的条件列
6.  W-Op：选择条件值的关系
7.  W-Val：选择的条件值
8. W-Rel：选择的条件类型

**3.2.4 模型集成**

load...

**3.2.5 损失函数**

根据不同的模型选择合适的损失函数，且总损失为各个子任务损失函数之和。

### 四、评价指标

评价指标包括：
**Logic Form Accuracy**：预测完全正确的SQL语句。其中，列的顺序并不影响准确率的计算。
**Execution Accuracy**：预测的SQL的执行结果与真实SQL的执行结果一致。
$$
Score_{lf}=\begin{cases} 1, SQL'=SQL \\ 0, SQL'\neq SQL \end{cases}
$$

$$
Acc_{lf} = {1\over N}\sum_{n=1}^N Score_{lf}^n
$$

$$
Score_{ex}=\begin{cases} 1, Y'=Y \\ 0, Y\neq Y\end{cases}
$$

$$
Acc_{ex} = {1\over N}\sum_{n=1}^N Score_{ex}^n
$$

其中, $N$表示数据量, $SQL′$和$SQL$分别代表预测的`SQL`语句和真实的`SQL`语句, $Score_{lf}$表示`Logic FormLogicForm`准确率; $Y′$和$Y$分别代表预测的`SQL`和真实的`SQL`的执行结果, $Score_{ex}$表示`Execution`准确率。

**Attention：**

* 1、`nn.CrossEntropyLoss()`是`nn.logSoftmax()`和`nn.NLLLoss()`的整合,可以直接使用它来替换网络中的这两个操作。
* 2、`NLLLoss()`的结果就是把上面的输出与`Label`对应的那个值拿出来，再去掉负号，再求均值

**五、工作计划：**

* 第一阶段：

  对现有的`baseline`模型进行代码分析，了解如何将整体任务分为数个子任务进行。`baseline`模型使用的是双向`lstm`模型，其主要工作是对问题和表头列名进行编码，编码后得到其语义信息，并通过交叉熵损失函数实现子任务的预测。其是在SQLnet的基础上进行改进的。

  [baseline参考地址](https://github.com/ZhuiyiTechnology/nl2sql_baseline)

  [SQLnet论文地址](https://arxiv.org/pdf/1711.04436.pdf)

* 第二阶段：

  `baseline`未加入任何数据预处理，可完成数据预处理和信息增强之后送入模型，并完成对比。

* 第三阶段：

  使用`bert代替双向的lstm`。

  1. 按照`baseline`的模式改成基于`bert`模型使用；

  2. 使用`X-SQL`。`X-SQL`使用改良版的`MT-DNN`，其主要分为六个子任务，但其是服务于`WiliSQL`英文数据集，在中文数据集上并不完全适配，需要改进。

     [X-SQL论文地址](https://www.microsoft.com/en-us/research/uploads/prod/2019/03/X_SQL-5c7db555d760f.pdf)

     [X-SQL解读](https://zhuanlan.zhihu.com/p/129244451)

