import flask
from flask import render_template
import os,sys
import io
from PIL import Image
import cv2

sys.path.append("../../train_test_mnist/")
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
                    predict_result,predict_probability = mnist.interface(image)
                    data["predict_result"] = str(predict_result[0])
                    data["predict_probability"] = str(predict_probability)

                    # indicate that the request was a success
                    data["success"] = True

        # return the data dictionary as a JSON response
        print(data)
    except Exception as e:
        log.error("{}".format(e))
    t2 = time.time()
    log.info("TIME :{} s".format(t2-t1))
    return flask.jsonify(data)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/webpage', methods=['POST', 'GET'])
def upload():
    if flask.request.method == 'POST':
        img = flask.request.files['img'].read()
        topk = flask.request.form['topk']
        image = Image.open(io.BytesIO(img))


        new_name = "./static/img/"+ time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + ".jpg"
        data = predict_img(image, topk=int(topk))
        image.save(new_name)
        return flask.jsonify({"result":data,"img_path":new_name})
    else:
        new_name = None
        result = []
    return render_template('upload.html', img_path=new_name, result=result)



def predict_img(img, topk=1):
    data = dict()
    start = time.time()
    predict_result,predict_probability = mnist.interface(img)
    result = [[int(predict_result[0]), predict_probability]]
    #add test data
    for i in range(1,topk):
        result.append([0,0])
    cost_time = time.time() - start
    data['predictions'] = list()
    for label, prob in result:
        m_predict = {'label': label, 'probability': ("%.4f" % prob)}
        data['predictions'].append(m_predict)
    data['state'] = True
    data['time'] = cost_time
    return data



if __name__ == "__main__":
    print(("* Loading tensorflow model and Flask starting server..."
        "please wait until server has fully started"))
    app.run('0.0.0.0', 5000)
