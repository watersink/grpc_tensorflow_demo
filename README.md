# web api Demo
web api demos using tensorflow,include grpc,flask,webpy,tornado,rabbitMQ,django,tf serving,tf cpp, ncnn ,mnn, openvino

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

    #freeze_optimize_quantize
    need source code compilation of tensorflow
    bazel build tensorflow/python/tools:freeze_graph
    bazel build tensorflow/python/tools:optimize_for_inference
    bazel build tensorflow/tools/quantization:quantize_graph
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

# freeze_optimize_quantize
    python3 0_graph_io.py
    bash 1_frozen_graph.sh
    bash 2_optimize_graph.sh
    bash 3_quantize_graph.sh
    python3 test_mnist_pb.py

# tensorflow-cpp-mnist
tested on tensorflow1.13,should build tensorflow from source [offical install](https://tensorflow.google.cn/install/source)

    bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    bazel build //tensorflow:libtensorflow_cc.so 


    cd tensorflow-cpp-mnist/cmake-build-debug
    cmake ..
    make
    ./main
    
    the results:
    Lenet init OK.
    1
    0.999875
    lenet Session Release...


# mnist_mnn
	python3 ckpt2pb.py
	/data/MNN/tools/converter/build/MNNConvert -f TF --modelFile mnist.pb --MNNModel mnn/mnist.mnn --bizCode MNN
	/data/MNN/tools/converter/build/MNNDump2Json.out mnn/mnist.mnn mnist.json
	/data/MNN/build/MNNV2Basic.out mnn/mnist.mnn 1 1
	/data/MNN_bak/build/benchmark.out mnn 100 0
	mkdir build&&cd build&&cmake ..&&make
	./mnist


    the results:
	4.74518e-07 4.74518e-07 4.74518e-07 4.74518e-07 0.999128 4.74518e-07 4.74518e-07 0.000137537 4.28663e-05 0.000688581


# mnist_ncnn
    python3 ckpt2pb.py
    tensorflow2ncnn ./mnist.pb ./ncnn/mnist.param ./ncnn/mnist.bin
    mkdir build&&cd build&&cmake ..&&make
    ./mnist


    the results:
	1.23689e-06 0.999917 5.75001e-06 1.23689e-06 2.05196e-05 1.23689e-06 3.1576e-05 1.23689e-06 1.88986e-05 1.23689e-06 


# mnist_openvino
    python3 python/ckpt2pb.py
    python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model mnist.pb -b 1
    #python
    python3 python/mnist.py
    python3 python/mnist_async.py
    #cpp
    cd cpp
    mkdir build&&cd build&&cmake ..&&make
    ./mnist
    ./mnist_async


    the results:
    classid    probability    label
    -------    -----------    -----
    0              4.74519e-07            0
    1              4.74519e-07            1
    2              4.74519e-07            2
    3              4.74519e-07            3
    4              0.999128            4
    5              4.74519e-07            5
    6              4.74519e-07            6
    7              0.000137548            7
    8              4.28666e-05            8
    9              0.000688585            9

    total inference time: 50.218
    Average running time of one iteration: 5.0218 ms

    Throughput: 199.132 FPS


    performance counts:

    conv2d/Conv2D                 EXECUTED       layerType: Convolution        realTime: 49         cpu: 49             execType: jit_avx2_FP32
    max_pooling2d/MaxPool         EXECUTED       layerType: Pooling            realTime: 27         cpu: 27             execType: jit_avx_FP32
    conv2d_2/Conv2D               EXECUTED       layerType: Convolution        realTime: 110        cpu: 110            execType: jit_avx2_FP32
    max_pooling2d_2/MaxPool       EXECUTED       layerType: Pooling            realTime: 16         cpu: 16             execType: jit_avx_FP32
    max_pooling2d_2/MaxPool_nC... EXECUTED       layerType: Reorder            realTime: 13         cpu: 13             execType: reorder_FP32
    Reshape                       NOT_RUN        layerType: Reshape            realTime: 0          cpu: 0              execType: unknown_FP32
    dense/MatMul                  EXECUTED       layerType: FullyConnected     realTime: 4670       cpu: 4670           execType: gemm_blas_FP32
    dense/Relu                    EXECUTED       layerType: ReLU               realTime: 9          cpu: 9              execType: jit_avx2_FP32
    dense_2/MatMul                EXECUTED       layerType: FullyConnected     realTime: 9          cpu: 9              execType: gemm_blas_FP32
    dense_2/Relu                  EXECUTED       layerType: ReLU               realTime: 6          cpu: 6              execType: jit_avx2_FP32
    output                        EXECUTED       layerType: SoftMax            realTime: 5          cpu: 5              execType: ref_any_FP32
    out_output                    NOT_RUN        layerType: Output             realTime: 0          cpu: 0              execType: unknown_FP32  




# reference
[https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/predict.proto](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/predict.proto)
> 
[https://grpc.io/docs/quickstart/python.html](https://grpc.io/docs/quickstart/python.html)
> 
[https://blog.keras.io/](https://blog.keras.io/)
> 
[https://www.tensorflow.org/serving/setup](https://www.tensorflow.org/serving/setup)
> 
[https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux)