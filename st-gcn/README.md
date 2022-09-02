## st-gcn

* 使用st-gcn完成`异常原因`的预测

### data

* 1、有`21个异常描述、120个异常原因、96个处理方法`，另`有4*2个KPI点`

* 2、在所有的故障小区清单中，一共有`4103`个不同的小区发生故障，小区发生故障次数`大于2`时有**329**个小区，当以`related_enb_id`为基准时，有**3578**个小区发生故障，`大于2`的小区数为**456**个。

  *注：单个小区发生故障大于2次保留*

* 3、共有26个时间点，某个小区在时间线上发生故障最多的次数为11次

### 项目结构

```python
.
├── README.md
├── config											# 训练和测试配置文件
│   ├── test.yaml
│   └── train.yaml
├── data												# 数据
├── feeder											# 数据导入
│   ├── feeder.py
│   └── tools.py
├── main.py											# 主函数
├── net													# 基本模型
│   ├── st_gcn.py
│   └── utils
│       ├── graph.py						# 根据neo4j数据库生成图数据
│       └── tgcn.py
├── processor										# 模型构建函数
│   ├── io.py
│   ├── prediction.py
│   └── processor.py
├── torchlight									# 模型gpu加速
│   ├── gpu.py
│   └── io.py
└── work_dir										# 保存模型、日志和评价指标
```

### 项目启动

未做`.sh`文件，`cd`到`st_gcn`下：`python main.py prediction -c ./config/train.yaml `
### Todo
* 1、项目服务
* 2、一些算法细节