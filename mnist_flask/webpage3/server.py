import flask
from flask import request , render_template
import os,sys
import io
from PIL import Image
import numpy as np
import hashlib
import time
import base64
import uuid
import cv2
import json
import requests
import functools

sys.path.append("../../train_test_mnist/")
from test_mnist import MNIST


import logging
from logging.handlers import TimedRotatingFileHandler,WatchedFileHandler
from logging.handlers import RotatingFileHandler
#日志打印格式
log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
formatter = logging.Formatter(log_fmt)
#创建TimedRotatingFileHandler对象
#log_file_handler = TimedRotatingFileHandler(filename="logs/attribute", when="midnight", interval=1, backupCount=1000)
log_file_handler = WatchedFileHandler(filename="logs/mnist")
log_file_handler.setFormatter(formatter)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()
#log.addHandler(log_file_handler)










mnist=MNIST()



app = flask.Flask(__name__)
@app.route("/predict", methods=["POST"])
def predict():
    t1 = time.time()
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    request_id=""
    md5=""

    try:
        # ensure an image was properly uploaded to our endpoint
        if flask.request.method == "POST":

            md5 = flask.request.form["md5"]
            request_id = flask.request.form["request_id"]
            data["md5"] = md5
            data["request_id"] = request_id

            if "image" in flask.request.files.keys():
                # read the image in PIL format
                bio = io.BytesIO()
                flask.request.files["image"].save(bio)
                file_bytes = bio.getvalue()
            elif "base64" in flask.request.form.keys():
                image_base64 = flask.request.form.get("base64")
                file_bytes = base64.urlsafe_b64decode(bytes(image_base64, encoding="utf-8"))
            elif "url" in flask.request.form.keys():
                image_url = flask.request.form["url"]
                response = requests.get(image_url)
                file_bytes = response.content

            else:
                log.error("request_id:{} field error, no field image".format(request_id))
                data["time_used"] = time.time() - t1
                return flask.jsonify(data)

            cal_md5sum = hashlib.md5(file_bytes).hexdigest()
            image = cv2.imdecode(np.frombuffer(file_bytes, dtype='uint8'), 1)
            image = cv2.resize(image,(28,28))

            if md5 == cal_md5sum:
                predict_result,predict_probability = mnist.interface(image)
                data["predict_result"] = str(predict_result[0])
                data["predict_probability"] = str(predict_probability)
                data["success"] = True
            else:
                log.error("request_id:{} md5 error, ori:{} now:{}".format(request_id, md5 ,cal_md5sum))


    except Exception as e:
        log.error("md5:{} request_id:{} exception:{}".format(md5, request_id, e))
        import traceback
        traceback.print_exc()

    
    t2 = time.time()
    log.info("TIME of {} id: {} :{} s".format(md5, request_id, t2-t1))
    data["time_used"] = t2-t1
    # return the data dictionary as a JSON response
    return flask.jsonify(data)




################
#web demo
################
@app.route('/get_message', methods=['POST'])
def get_message():
    cmd = request.form['message_cmd']
    r = os.popen('/bin/bash -c "{}"'.format(cmd))
    text = r.read()
    r.close()
    return text



@app.route('/webpage', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/webpage', methods=['POST'])
def index_post():

    t1 = time.time()
    data = {"errorCode": 0,
        "errorMsg": "",
        "time_used": 0,
        "md5": "",
    }
    md5 = ""
    img_stream = ""
    try:
        bio = io.BytesIO()
        flask.request.files["file"].save(bio)
        file_bytes = bio.getvalue()

        md5 = hashlib.md5(file_bytes).hexdigest()
        image = cv2.imdecode(np.frombuffer(file_bytes, dtype='uint8'), 1)



        predict_result,predict_probability = mnist.interface(image)
        data["predict_result"] = str(predict_result[0])
        data["predict_probability"] = str(predict_probability)

        data["md5"] = md5

        img_bytes = cv2.imencode(".jpg", image)[1].tobytes()
        img_stream = base64.b64encode(img_bytes).decode()


    except Exception as e:
        log.error("md5:{} exception:{}".format(md5, e))
        data["errorCode"] = 3
        data["errorMsg"] = "md5:{} exception:{}".format(md5, e)
        import traceback
        traceback.print_exc()

    
    t2 = time.time()
    log.info("TIME of md5: {} :{} s".format(md5, t2-t1))
    data["time_used"] = t2-t1



    return render_template('index.html',  output_json = json.dumps(data), img_stream=img_stream)




#支持，自动归档，多进程，进程守护
if __name__ == '__main__':
    import logging
    app.logger.setLevel(logging.DEBUG)
    app.run(host="0.0.0.0", port=5000)

