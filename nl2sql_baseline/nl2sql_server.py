# -*- coding:utf-8 -*-
# @Time: 2021/2/22 20:48
# @File: nl2sql_server.py.py
# @Software: PyCharm
# @Author: xiehl
# -------------------------
import traceback
from test import *
from config import *
from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequestKeyError
import json
import platform
from logs.log import *

import time
t = time.time()
app_company_loss = Flask('app')

# 单税种
@app_company_loss.route('/nl2sql', methods=['POST'])
def apicall():
    try:
        start = time.time()
        req_data = request.get_data(as_text=True)
        logger.info(req_data)
        if req_data:
            req_data = json.loads(req_data)

            test_sql_data = req_data['question']
            test_table_id = req_data['table_id']

            dict_pred = main(test_sql_data=test_sql_data, test_table_id=test_table_id)

            logger.info("时间{}".format(time.time() - start))
            dict_ = {}
            dict_['status'] = 'success'
            dict_['sql'] = dict_pred

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
        logger.info(traceback.format_exc())
        res = {'status': 'failed', 'results': str(e)}
        return jsonify(res)

@app_company_loss.errorhandler(400)
def bad_request(error=None):
    message = {
        'status': 400,
        'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
    }
    resp = jsonify(message)
    resp.status_code = 400
    return resp



if __name__ == "__main__":

    logger.info('<<<nl2sql Server Started')
    sysstr = platform.system()
    logger.info(sysstr)
    if (sysstr == "Windows"):
        app_company_loss.run(host="0.0.0.0", port=int(server_port), threaded=False, debug=True)
    elif (sysstr == "Linux"):
        app_company_loss.run(host="0.0.0.0", port=int(server_port), threaded=True)
    else:
        app_company_loss.run(host="0.0.0.0", port=int(server_port), threaded=True)
    logger.info('<<<nl2sql Server stopped')