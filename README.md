# web api Demo
web api demos using tensorflow,include grpc,flask,webbpy,tornodo,tf serving

# install
    #grpc
    pip3 install grpcio
    pip3 install protobuf
    pip3 install grpcio-tools
    #flask
    pip3 install flask
  
# helloworld
    python3 -m grpc_tools.protoc -I ./ –-python_out=./ –-grpc_python_out=./ ./helloworld.proto
    python3 server.py
    python3 client.py
  
# mnist_tf_network
    python3 -m grpc_tools.protoc -I ./ –-python_out=./ –-grpc_python_out=./ ./mnist.proto
    python3 server.py
    python3 client.py
# mnist_tf_interface
    python3 -m grpc_tools.protoc -I ./ –-python_out=./ –-grpc_python_out=./ ./mnist.proto
    python3 server.py
    python3 client.py

# mnist_flask
    python server.py
    python3 client.py
    sh curl.sh
# reference
[https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/predict.proto](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/predict.proto)
[https://grpc.io/docs/quickstart/python.html](https://grpc.io/docs/quickstart/python.html)
[https://blog.keras.io/](https://blog.keras.io/)