# -*- coding: utf-8 -*-
import grpc
import time
from concurrent import futures 
import helloworld_pb2, helloworld_pb2_grpc

_HOST = 'localhost'
_PORT = '8088'

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class gRPCServicer_hello(helloworld_pb2_grpc.helloworldServicer):

    def sayhello(self, request, context):
        print ("called with " + request.name)
        return helloworld_pb2.HelloReply(message='Hello, %s!' % request.name)


def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  helloworld_pb2_grpc.add_helloworldServicer_to_server(gRPCServicer_hello(), server)
  server.add_insecure_port('[::]:'+_PORT)
  server.start()
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)

if __name__ == '__main__':
    serve()