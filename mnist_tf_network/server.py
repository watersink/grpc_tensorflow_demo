#!/usr/bin/env python
# -*- coding: utf-8 -*-

from concurrent import futures
import time
import json
import grpc
import numpy as np
import tensorflow as tf
import logging
from tensorflow.python.framework import tensor_util

import mnist_pb2,mnist_pb2_grpc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.DEBUG)

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = 'localhost'
_PORT = '8088'

class MnistPredictionServiceSubclass(mnist_pb2_grpc.MnistPredictionServiceServicer):
    def __init__(self):
        self.checkpoint_file = "../train_test_mnist/save/"
        self.graph_file = "../train_test_mnist/save/mnist.ckpt.meta"

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()

            # Restore graph and weights from the model file
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_file)
            if ckpt and ckpt.model_checkpoint_path:
                logging.info("Use the model: {}".format(ckpt.model_checkpoint_path))
                saver = tf.train.import_meta_graph(self.graph_file)
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                self.inputs = json.loads(tf.get_collection('input')[0])
                self.outputs = json.loads(tf.get_collection('output')[0])
            else:
                logging.error("No model found, exit")
                exit()

    def MnistPredict(self, request, context):
        request_map = request.inputs
        feed_dict = {}
        for k, v in self.inputs.items():
            # Convert TensorProto objects to numpy
            feed_dict[v] = tensor_util.MakeNdarray(request_map[k])

        # Example result: {'key': array([ 2.,  2.], dtype=float32), 'prediction': array([1, 1]), 'softmax': array([[ 0.07951042,  0.92048955], [ 0.07951042,  0.92048955]], dtype=float32)}
        predict_result = self.sess.run(self.outputs, feed_dict=feed_dict)
        logging.info("result{}".format(predict_result))

        response = mnist_pb2.MnistPredictResponse()
        for k, v in predict_result.items():
            # Convert numpy objects to TensorProto
            response.outputs[k].CopyFrom(tensor_util.make_tensor_proto(v))
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
