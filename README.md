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

    #tf serving
    echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
    sudo apt-get update && sudo apt-get install tensorflow-model-server
    sudo apt-get upgrade tensorflow-model-serve
    pip3 install tensorflow-serving-api

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

# mnist_tfServing
    python3 export_mnist.py
    tensorflow_model_server --port=8500 --model_name=mnist --model_base_path=/opt/grpc_tensorflow_demo/mnist_tfServing/mnist
    python3 client.py

# reference
[https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/predict.proto](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/predict.proto)
[https://grpc.io/docs/quickstart/python.html](https://grpc.io/docs/quickstart/python.html)
[https://blog.keras.io/](https://blog.keras.io/)
[https://www.tensorflow.org/serving/setup](https://www.tensorflow.org/serving/setup)