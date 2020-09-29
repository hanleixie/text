import os
# 获取当前文件路径
current_path = os.path.abspath(__file__)
# 获取当前文件的父目录
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")

import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append(father_path)
# sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequestKeyError
import json
import platform
import logging as log
#from pred_main import *
from model_builder_one import *
from one_predict.config import *
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"    # 日志格式化输出
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"                        # 日期格式
fp = log.FileHandler('server_log.txt', encoding='utf-8')
fs = log.StreamHandler()
log.basicConfig(level=log.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=[fp, fs])
import time
t = time.time()
app = Flask('app')


@app.route('/bertsumabs', methods=['POST'])
def apicall():
    log.info('<<bertsumabs')
    try:
        start = time.time()
        req_data = request.get_data(as_text=True)
        log.info(req_data)
        if req_data:
            req_data = json.loads(req_data)
            doc = req_data['doc']

            dict_res = bertsumabs_predict(doc)

            log.info("时间{}".format(time.time() - start))
            dict_ = {}
            dict_['status'] = 'success'
            dict_['results'] = dict_res
            return jsonify(dict_)
        else:
            res = {'status': 'failed', 'results': '没有收到request消息'}
            return jsonify(res)
    except BadRequestKeyError as e:
        log.error(e)
        res = {'status': 'failed', 'results': str(e)}
        return jsonify(res)
    except FileNotFoundError as e:
        log.error(e)
        res = {'status': 'failed', 'results': e.strerror}
        return jsonify(res)
    except Exception as e:
        log.error(e)
        res = {'status': 'failed', 'results': str(e)}
        return jsonify(res)




@app.errorhandler(400)
def bad_request(error=None):
    message = {
        'status': 400,
        'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
    }
    resp = jsonify(message)
    resp.status_code = 400
    return resp



if __name__ == "__main__":

    log.info('<<<Ocr Server Started')
    sysstr = platform.system()
    log.info(sysstr)
    if (sysstr == "Windows"):
        app.run(host="0.0.0.0", port=int(server_port), threaded=False)
    elif (sysstr == "Linux"):
        app.run(host="0.0.0.0", port=int(server_port), threaded=True)
    else:
        app.run(host="0.0.0.0", port=int(server_port), threaded=True)
    log.info('<<<Ocr Server stopped')







