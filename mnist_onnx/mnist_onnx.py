import numpy as np
import cv2
import datetime

import torch
import onnxruntime
import onnx

class MNIST(object):
    def __init__(self, model_path = "mnist.onnx"):
        self.session = onnxruntime.InferenceSession(model_path)
        self.inputs = self.session.get_inputs()[0].name


    def inference(self, img):

        begin = datetime.datetime.now()
        image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        image =cv2.resize(image,(28, 28))
        image = image/255
        input_image = np.expand_dims(np.expand_dims(image,0),0).astype(np.float32)

        output = self.session.run(None, {self.inputs: input_image})



        end = datetime.datetime.now()
        print("cpu times = ", end - begin)
        return output 


if __name__ == '__main__':
    image=cv2.imread("../train_test_mnist/MNIST/testimage/5/1.jpg",1)
    mnist = MNIST()
    output = mnist.inference(image)
    print(output)
