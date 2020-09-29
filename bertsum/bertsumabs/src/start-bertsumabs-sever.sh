#!/bin/bash
# ----------------------------------------------------------------------
# name:         start-bertsumabs-sever.sh
# version:      1.0
# createTime:   2020-09-28
# description:  ocr server鍚姩鑴氭湰
# author:       wangjianhua
# email:        wangjhr@yonyou.com
# ----------------------------------------------------------------------

OcrMainDir='/home/yrobot/tmp/code/bertsumabs/src'
CondaEnvirName='bert_sum'

cd $OcrMainDir

source deactivate
source activate $CondaEnvirName

if [ $1 == "start" ];then
nohup python bertsumabs_server.py 1>bertsumabs_server.log 2>&1 &
echo "$!" > pid
elif [ $1 == "stop" ];then
kill `cat pid`
else
echo "Please make sure the position variable is start or stop."
fi