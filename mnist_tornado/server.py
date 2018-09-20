import tornado.ioloop
import tornado.web


import os,sys
import io
import base64
import numpy as np
import json
import cv2

sys.path.append("../train_test_mnist/")
from test_mnist import MNIST


mnist=MNIST()


class MNIST_SERVER_Handler(tornado.web.RequestHandler):
    def get(self):
        pass

    def post(self):
        try:
            image_get = self.get_argument('image')
        except ValueError as e:
            return e
        
        data = {"success": False}
        image_file_value = base64.urlsafe_b64decode(image_get)
        image = cv2.imdecode(np.frombuffer(image_file_value, dtype='uint8'), 1)

        predict_result,predict_probability =mnist.interface(image)
        print("{}{}".format(predict_result,predict_probability ))
        data["predict_result"] = str(predict_result[0])
        data["predict_probability"] = str(predict_probability)

        data["success"] = True

        return_json_str=json.dumps(data, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)

        self.write(return_json_str)
        self.finish()



if __name__ == "__main__":
    print(("* Loading tensorflow model and webpy starting server..."
        "please wait until server has fully started"))
    app = tornado.web.Application([(r'/predict', MNIST_SERVER_Handler)])
    app.listen(8080)
    tornado.ioloop.IOLoop.instance().start()

