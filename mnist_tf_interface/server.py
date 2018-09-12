#!/usr/bin/env python
# -*- coding: utf-8 -*-

from concurrent import futures
import time
import json
import grpc
import numpy as np
import tensorflow as tf
import logging
import cv2

import mnist_pb2,mnist_pb2_grpc
import os,sys
sys.path.append("../train_test_mnist/")
from test_mnist import MNIST
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.DEBUG)

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = 'localhost'
_PORT = '8088'

class MnistPredictionServiceSubclass(mnist_pb2_grpc.MnistPredictionServiceServicer):
    def __init__(self):
        self.mnist=MNIST()

    def MnistPredict(self, request, context):
        request_map = request.inputs
        inputs = cv2.imdecode(np.frombuffer(request.inputs, dtype='uint8'), 1)

        predict_result,predict_probability=self.mnist.interface(inputs)
        logging.info("predict_result{}predict_probability{}".format(predict_result,predict_probability))
        response = mnist_pb2.MnistPredictResponse()
        response.outputs=str(predict_result[0])
        response.probability=predict_probability
        return response


def serve():
    """Start the gRPC service."""
    prediction_service = MnistPredictionServiceSubclass()
    logging.info("Start gRPC server with PredictionService: {}".format(vars(
        prediction_service)))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    mnist_pb2_grpc.add_MnistPredictionServiceServicer_to_server(MnistPredictionServiceSubclass(), server)
    server.add_insecure_port('[::]:'+_PORT)
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
