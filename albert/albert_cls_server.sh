#!/bin/bash
# ----------------------------------------------------------------------
# name:         albert_cls_server.sh
# version:      1.0
# createTime:   2020-12-10
# description:  start albert_cls_server
# author:       xiehl
# email:        xiehlb@yonyou.com
# ----------------------------------------------------------------------

export PYTHONPATH=$(pwd)

OcrMainDir='/home/yrobot/xiehanlei/albert_pytorch'
cd $OcrMainDir

PYTHON_HOME=$(conda env list | grep albert_cls | grep -v grep | awk '{print $2}')

echo $PYTHON_HOME

if [ $1 == "start" ];then
   nohup $PYTHON_HOME/bin/python $OcrMainDir/emotion_server.py 1>server_test.log 2>&1 &
   echo "$!" > pid
elif [ $1 == "stop" ];then
   kill `cat pid`
else
   echo "Please make sure the position variable is start or stop."
fi