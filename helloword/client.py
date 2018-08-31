# -*- coding: utf-8 -*-
"""The Python implementation of the gRPC client."""
from __future__ import print_function
import grpc
import helloworld_pb2, helloworld_pb2_grpc 
_HOST = 'localhost'
_PORT = '8088'

def run():
    conn = grpc.insecure_channel(_HOST + ':' + _PORT)
    client = helloworld_pb2_grpc.helloworldStub(channel=conn)
    response = client.sayhello(helloworld_pb2.HelloRequest(name='David'))
    print("received: " + response.message)

if __name__ == '__main__':
    run()