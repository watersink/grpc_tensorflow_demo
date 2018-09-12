#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import grpc
import json
import numpy as np
import cv2
from tensorflow.python.framework import tensor_util

import mnist_pb2,mnist_pb2_grpc
_HOST = 'localhost'
_PORT = '8088'

def main():
    # Connect with the gRPC server
    server_address = _HOST+":"+_PORT
    request_timeout = 5.0
    channel = grpc.insecure_channel(server_address)
    stub = mnist_pb2_grpc.MnistPredictionServiceStub(channel)


    request = mnist_pb2.MnistPredictRequest()
    image = cv2.imread("../train_test_mnist/MNIST/testimage/5/1.jpg")
    img_bytes = cv2.imencode('.png', image)[1].tobytes()
    request.inputs=img_bytes

    response = stub.MnistPredict(request, request_timeout)

    predict_result=response.outputs
    predict_probability=response.probability
    print(predict_result)
    print(predict_probability)


if __name__ == '__main__':
    main()
