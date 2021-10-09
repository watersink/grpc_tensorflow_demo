import flask
import os,sys
import io
from PIL import Image
import cv2
from flask import Flask, request, render_template, session, flash, redirect, url_for, jsonify
from celery import Celery
import hashlib
import datetime
import requests
import json
import codecs

sys.path.append("../train_test_mnist/")
from test_mnist import MNIST
import hashlib




import time
import logging
from logging.handlers import TimedRotatingFileHandler,WatchedFileHandler
from logging.handlers import RotatingFileHandler
#日志打印格式
log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
formatter = logging.Formatter(log_fmt)
#创建TimedRotatingFileHandler对象
#log_file_handler = TimedRotatingFileHandler(filename="mnist", when="midnight", interval=1, backupCount=1000)
log_file_handler = WatchedFileHandler(filename="mnist")
log_file_handler.setFormatter(formatter)
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()
log.addHandler(log_file_handler)





mnist=MNIST()



app = flask.Flask(__name__)
#app.config['SECRET_KEY'] = 'top-secret!'

redis_ip = 'redis://127.0.0.1:6379/0' #test environment

  
# 配置
# 配置消息代理的路径，如果是在远程服务器上，则配置远程服务器中redis的URL
app.config['CELERY_BROKER_URL'] = redis_ip
# 要存储 Celery 任务的状态或运行结果时就必须要配置
app.config['CELERY_RESULT_BACKEND'] = redis_ip
# 初始化Celery
celery = Celery(app.name, broker = redis_ip)
# 将Flask中的配置直接传递给Celery
celery.conf.update(app.config)




# bind为True，会传入self给被装饰的方法
@celery.task(bind=True)
def process_image(self, file_bytes_str):
    file_bytes = bytes(file_bytes_str[2:-1], encoding="utf-8")
    file_bytes = codecs.escape_decode(file_bytes, "hex-escape")[0]
    json_values={'md5':hashlib.md5(file_bytes).hexdigest()}
    json_files = {'image':file_bytes}

    out_json = requests.post("http://127.0.0.1:5000/predict", data=json_values,files=json_files).json()
    

    total_num = 0
    self.update_state(state='PROGRESS', meta={'current': total_num, 'total': total_num, 'status': "Processing"})

    out_json = {"status": "Task completed!", "result": out_json}
    # 返回字典
    return out_json



@app.route('/predict_asy', methods=['POST'])
def predict_asy():
    t1 = time.time()
    dt = datetime.datetime.now()

    out_data = {"time_used": 0, "location":"", "errorCode":0, "errorMsg":"", "location":""}
    post_id = ""
    task_id = ""
    try:
        if flask.request.files.get("image"):
            file_bytes = flask.request.files["image"].read()
        else:
            log.error("date:{}, error:no image field".format(dt))
            out_data["errorCode"] = 1
            out_data["errorMsg"] = "no image field"
            out_data["time_used"] = time.time() - t1
            return jsonify(out_data)


        # 异步调用
        task = process_image.apply_async(args=[str(file_bytes)])
        task_id = task.id
        # 返回 202，与Location头
        out_data["location"] = url_for('taskstatus',task_id=task.id)
    except Exception as e:
        log.error("date:{} post_id:{} exception:{}".format(dt, post_id, e))
        out_data["errorCode"] = 2
        out_data["errorMsg"] = "post_id:{} exception:{}".format(post_id, e)
        import traceback
        traceback.print_exc()



    t2 = time.time()
    log.info("date:{} post_id:{} time cost:{} s".format(dt, post_id, t2-t1))
    out_data["time_used"] = t2-t1

    return jsonify(out_data), 202, {'Location': url_for('taskstatus',task_id = task_id)}



@app.route('/status/<task_id>')
def taskstatus(task_id):
    task = process_image.AsyncResult(task_id)
    if task.state == 'PENDING': # 在等待
        response = {
        'state': task.state,
        'current': 0,
        'total': 1,
        'status': 'Pending...',
        }
    elif task.state != 'FAILURE': # 没有失败
        response = {
        'state': task.state, # 状态
        # meta中的数据，通过task.info.get可以获得
        'current': task.info.get('current', 0), # 当前循环进度
        'total': task.info.get('total', 1), # 总循环进度
        'status': task.info.get('status', ''),
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
        if 'post_id' in task.info:
            response['post_id'] = task.info['post_id']
    else:
        # 后端执行任务出现了一些问题
        response = {
        'state': task.state,
        'current': 1,
        'total': 1,
        'status': str(task.info), # 报错的具体异常
        }
    return jsonify(response)






@app.route("/predict", methods=["POST"])
def predict():
    t1 = time.time()
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    try:
        # ensure an image was properly uploaded to our endpoint
        if flask.request.method == "POST":
            md5=flask.request.form['md5']
            if flask.request.files.get("image"):
                # read the image in PIL format
                file_bytes = flask.request.files["image"].read()
                cal_md5sum = hashlib.md5(file_bytes).hexdigest()
                image = Image.open(io.BytesIO(file_bytes))


                # classify the input image and then initialize the list
                # of predictions to return to the client
                if md5 == cal_md5sum:
                    print("#####")
                    predict_result,predict_probability = mnist.interface(image)
                    data["predict_result"] = str(predict_result[0])
                    data["predict_probability"] = str(predict_probability)

                    print("#####", predict_result,predict_probability)
                    # indicate that the request was a success
                    data["success"] = True

        # return the data dictionary as a JSON response
        print(data)
    except Exception as e:
        log.error("{}".format(e))
    t2 = time.time()
    log.info("TIME :{} s".format(t2-t1))
    return flask.jsonify(data)




if __name__ == "__main__":
    print(("* Loading tensorflow model and Flask starting server..."
        "please wait until server has fully started"))
    app.run('0.0.0.0', 5000)
