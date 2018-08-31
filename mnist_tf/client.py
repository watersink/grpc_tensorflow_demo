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

    # Make request data
    request = mnist_pb2.MnistPredictRequest()

    image = cv2.imread("./MNIST/testimage/5/1.jpg")
    array = np.array(image)/(255*1.0)-0.5
    samples_features =  array.reshape([-1,28,28,3])

    # samples_features = np.array(
    #     [[10, 10, 10, 8, 6, 1, 8, 9, 1], [10, 10, 10, 8, 6, 1, 8, 9, 1]])
    samples_keys = np.array([1])
    # Convert numpy to TensorProto
    request.inputs["features"].CopyFrom(tensor_util.make_tensor_proto(
        samples_features))
    request.inputs["key"].CopyFrom(tensor_util.make_tensor_proto(samples_keys))

    # Invoke gRPC request
    response = stub.MnistPredict(request, request_timeout)

    # Convert TensorProto to numpy
    result = {}
    for k, v in response.outputs.items():
        result[k] = tensor_util.MakeNdarray(v)
    print(result["softmax"][0])
    print(result["prediction"])


if __name__ == '__main__':
    main()
