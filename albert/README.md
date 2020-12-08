# Albert的情感分类

## 目的

通过Albert完成情感二分类

## 数据格式和评价指标

* 1、数据来源：

  数据为对车的评价，如下所示：

  `外观漂亮，空间够大，动力家用也ok`；`轮胎挡板隔音不太好，小石子弹起来的时候声音有点大。`第一句话标签为正类1，第二句话标签为负类0。

* 2、数据格式：

  第一行为`label`和`text`，每行包括三列，第一列为样本编号，第二列和第三列为`label`和`text`。

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
├── scripts											  #运行脚本
│   └── run_classifier_car.sh
└── tools												 #基础函数
    └── common.py

## 调用方式

* 在`config.py`中修改配置参数，调用`run_classifier.py`。
* 当想修改数据文件文字和分类数时，在`glue.py`中进行修改。