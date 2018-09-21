# web api Demo
web api demos using tensorflow,include grpc,flask,webpy,tornado,rabbitMQ,django,tf serving

# install
    #grpc
    pip3 install grpcio
    pip3 install protobuf
    pip3 install grpcio-tools
    
    #flask
    pip3 install flask
    
    #webpy
    git clone https://github.com/webpy/webpy.git
    python3 setup.py install
    pip3 install requests
    
    #tornado
    pip3 install tornado

    #rabbitMQ
    pip3 install pika

    #Django
    pip3 install django
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
    python3 server.py
    python3 client.py
    sh curl.sh

# mnist_webpy
    python3 server.py
    python3 client.py
    sh curl.sh

# mnits_tornado
    python3 server.py
    python3 client.py

# mnist_rabbitMQ
    python3 server.py
    python3 client.py

# mnist_Django
    django-admin startproject mnist_Django
    python3 manage.py runserver
    python3 client.py
# reference
[https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/predict.proto](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/predict.proto)
[https://grpc.io/docs/quickstart/python.html](https://grpc.io/docs/quickstart/python.html)
[https://blog.keras.io/](https://blog.keras.io/)