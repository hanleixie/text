# -*- coding:utf-8 -*-
# @Time: 2020/12/10 9:46
# @File: emotion_server.py
# @Software: PyCharm
# @Author: xiehl
# -------------------------
from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequestKeyError
import json
import platform
from run_classifier import for_server
from config import server_port
from tools.common import init_logger, logger
init_logger()
import time
t = time.time()
app = Flask('app')

@app.route('/albert_cls', methods=['POST'])
def apicall():
    logger.info('<<albert_cls')
    try:
        start = time.time()
        req_data = request.get_data(as_text=True)
        logger.info(req_data)
        if req_data:
            req_data = json.loads(req_data)
            text = req_data['text']
            task = req_data['task']

            dict_res = for_server(text=text, task=task)

            logger.info("时间{}".format(time.time() - start))
            dict_ = {}
            dict_['status'] = 'success'
            dict_['results'] = dict_res
            return jsonify(dict_)
        else:
            res = {'status': 'failed', 'results': '没有收到request消息'}
            return jsonify(res)
    except BadRequestKeyError as e:
        logger.error(e)
        res = {'status': 'failed', 'results': str(e)}
        return jsonify(res)
    except FileNotFoundError as e:
        logger.error(e)
        res = {'status': 'failed', 'results': e.strerror}
        return jsonify(res)
    except Exception as e:
        logger.error(e)
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

    logger.info('<<<Ocr Server Started')
    sysstr = platform.system()
    logger.info(sysstr)
    if (sysstr == "Windows"):
        app.run(host="0.0.0.0", port=int(server_port), threaded=False)
    elif (sysstr == "Linux"):
        app.run(host="0.0.0.0", port=int(server_port), threaded=True)
    else:
        app.run(host="0.0.0.0", port=int(server_port), threaded=True)
    logger.info('<<<Ocr Server stopped')







