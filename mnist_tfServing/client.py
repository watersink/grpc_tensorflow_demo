import sys
import cv2
import numpy as np
import tensorflow as tf
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util


class MNIST_CLIENT(object):
    def __init__(self):
        self.url_port="127.0.0.1:8500"
        self.name="mnist"
        self.signature_name='predict_images'
    def process(self,image):
        array = np.array(image)/(255*1.0)-0.5
        samples_features =  array.reshape([-1,28,28,3])
       

        channel = grpc.insecure_channel(self.url_port)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.name
        request.model_spec.signature_name = self.signature_name
        request.inputs['image'].CopyFrom(tf.contrib.util.make_tensor_proto(samples_features, dtype=dtypes.float32))
        result_future = stub.Predict.future(request, 5.0)
        response=result_future.result().outputs



        result = {}
        for k, v in response.items():
            result[k] = tensor_util.MakeNdarray(v)
        return result

if __name__ == '__main__':
    image = cv2.imread("../train_test_mnist/MNIST/testimage/5/1.jpg")
    mnist_client=MNIST_CLIENT()
    result=mnist_client.process(image)
    print(result["result"][0])
    print(result["probability"])



