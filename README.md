# gRPC Demo

# install
    pip install grpcio
    pip install protobuf
    pip install grpcio-tools
  
# helloworld
    python -m grpc_tools.protoc -I ./ –-python_out=./ –-grpc_python_out=./ ./helloworld.proto
    python server.py
    python client.py
  
# mnist_tf
    python -m grpc_tools.protoc -I ./ –-python_out=./ –-grpc_python_out=./ ./mnist.proto
    python server.py
    python client.py

# reference
[https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/predict.proto](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/predict.proto)