import web
import os,sys
import io
import base64
import numpy as np
import json
import cv2

sys.path.append("../train_test_mnist/")
from test_mnist import MNIST


mnist=MNIST()

urls=('/predict','MNIST_SERVER')

class MNIST_SERVER():
    def GET(self):
        pass
    def POST(self):
        try:
            form = web.input()
        except ValueError as e:
            return e
        
        data = {"success": False}
        image_file_value = base64.urlsafe_b64decode(form['image'])
        image = cv2.imdecode(np.frombuffer(image_file_value, dtype='uint8'), 1)

        predict_result,predict_probability = mnist.interface(image)
        data["predict_result"] = str(predict_result[0])
        data["predict_probability"] = str(predict_probability)

        data["success"] = True

        return_json_str=json.dumps(data, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)
        web.header('Content-Type', 'application/json')
        web.header('Content-Length', len(return_json_str.encode('utf-8')))
        return return_json_str

if __name__ == "__main__":
    print(("* Loading tensorflow model and webpy starting server..."
        "please wait until server has fully started"))
    app = web.application(urls, globals())
    app.run()