# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import mnist_pb2 as mnist__pb2


class MnistPredictionServiceStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.MnistPredict = channel.unary_unary(
        '/mnist.MnistPredictionService/MnistPredict',
        request_serializer=mnist__pb2.MnistPredictRequest.SerializeToString,
        response_deserializer=mnist__pb2.MnistPredictResponse.FromString,
        )


class MnistPredictionServiceServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def MnistPredict(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_MnistPredictionServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'MnistPredict': grpc.unary_unary_rpc_method_handler(
          servicer.MnistPredict,
          request_deserializer=mnist__pb2.MnistPredictRequest.FromString,
          response_serializer=mnist__pb2.MnistPredictResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'mnist.MnistPredictionService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
