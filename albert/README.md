# 基于Albert的分类任务

## 目的

基于Albert的分类任务，给定训练语料和标签，完成对对测试语料的预测。

## 数据格式和评价指标

* 1、数据样例：

  数据为对车的评价，如下所示：

  `外观漂亮，空间够大，动力家用也ok`；`轮胎挡板隔音不太好，小石子弹起来的时候声音有点大。`第一句话标签为正类1，第二句话标签为负类0。

* 2、数据格式：

  第一行为`label`和`text`，每行包括三列，第一列为样本编号，第二列和第三列为`label`和`text`。

  ```
  	label	text
  0	1	操控性舒服、油耗低，性价比高
  1	0	动力的确有点点让我相信了up的确是个代步车而已!
  2	1	1。车的外观很喜欢。2。省油，现在磨合期7.3，相信以后还会下降。
  3	1	内饰的做工和用料同级别同价位最厚道的
  4	0	减震系统太硬！
  5	0	售后不好，4s店不全，看来以后要自己保养了，而且4s的态度也不够好
  ```

* 3、评价指标

  评价指标为`acc`和`f1`，准确度和`f1`值。

## 项目架构

    .
    ├── callback                                               #损失函数和进程显示
    │   ├── lr_scheduler.py
    │   ├── optimization
    │   │   ├── adamw.py
    │   └── progressbar.py
    ├── dataset												#数据
    ├── metrics												#评价
    │   ├── compute_metrics.py
    ├── model												 #Albert相关模型
    │   ├── configuration_albert.py
    │   ├── configuration_utils.py
    │   ├── file_utils.py
    │   ├── modeling_albert.py
    │   ├── modeling_utils.py
    │   ├── tokenization_albert.py
    │   └── tokenization_utils.py
    ├── outputs											 #输出
    ├── prev_trained_model						 #预训练模型
    ├── processors									   #数据处理
    │   ├── glue.py
    │   └── utils.py
    ├── requirements.txt							   #需求模块
    ├── run_classifier.py							   #运行函数
    ├── config.py							   #配置文件
    ├── emotion_server.py							   #运行函数
    ├── scripts											  #运行脚本
    │   └── run_classifier_car.sh
    └── tools												 #基础函数
        └── common.py

## 调用方式

* 在`config.py`中修改配置参数，调用`run_classifier.py`。

* 当想修改数据文件文字和分类数时，在`config.py`中进行修改；在`tasks_num_labels`中修改 labels 和 tasks。

  ```
  tasks_num_labels = {
      'car': ['不好', '好'],
      'news': ['金融', '民生', '房地产', '教育', '科技', '法律', '国际', '运动', '游戏', '娱乐']
  }
  'news'对应的标签分别为各自的索引，如'金融'的标签为'0'，娱乐的标签为'9'
  ```

  在`processors`中修改 task 和 语料。

  ```
  processors = {
      'car': ['train_1.tsv', 'dev_1.tsv', 'test_1.tsv'],
      'news': ['train_1.txt', 'dev_1.txt', 'test_1.txt']
  }
  ```

  在`output_modes`中修改 task 和 task_cls_type。

  ```
  output_modes = {
      'car': "emotion_classification",
      'news': "news_classfification",
  }
  ```

## 调用服务

* Windows下调用`emotion_server.py`、服务器运行`albert_cls_server.sh`，`port=8201`。

* postman自测：Windows `http://0.0.0.0:8201/albert_cls`、服务器`http://10.10.65.251:8201/albert_cls`。

* 输入格式：

  ```
  {"text" : "今年高考山东卷考试内容比较难。一本线会较以往年份偏低", "task":"news"}
  ```

* 输出：

  ```
  {
      "results": "教育",
      "status": "success"
  }
  ```

  

