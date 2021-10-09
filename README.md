# web api Demo
web api demos using tensorflow,include grpc,flask,webpy,tornado,rabbitMQ,redis,celery,django,tf serving,tf cpp, tflite, ncnn ,mnn, openvino, movidius_ncs, libtorch , onnxruntime, c++ crow http server, c++ libcurl client, triton_inference_server

# install
    #grpc
    pip3 install grpcio
    pip3 install protobuf
    pip3 install grpcio-tools
    
    #flask
    pip3 install flask
    pip3 install gunicorn

    #flask_asynchronous
	pip3 install celery==4.4.7
	pip3 install redis==3.5.3

    
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


    #onnxruntime
    #cpu
    pip3 install onnxruntime
    #gpu
    pip3 install onnxruntime-gpu

    #mnist_cpp_http
    apt-get install libboost-all-dev
    
    #mnist_triton_inference_server
    pip3 install tensorrtserver-1.12.0-py3-none-linux_x86_64.whl
    pip3 install tritongrpcclient-1.12.0-py3-none-linux_x86_64.whl


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
## webpage1
![](mnist_flask/webpage1/static/img/webpage1.png)
## webpage2
![](mnist_flask/webpage2/static/results/webpage2.png)
## webpage3
![](mnist_flask/webpage3/logs/webpage3.png)



# mnist_flask_asynchronous
    redis-server &
    celery worker -A app.celery --loglevel=info &
    python3 server.py
    python3 client.py
    


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


# tflite
    bash to_tflite.sh
    #python
    python3 mnist_tflite.py
    the results:
    [{'name': 'input', 'index': 12, 'shape': array([ 1, 28, 28,  3], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]
    [{'name': 'output', 'index': 15, 'shape': array([ 1, 10], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]
    results:[4.7451897e-07 4.7451897e-07 4.7451897e-07 4.7451897e-07 9.9912816e-01
    4.7451897e-07 4.7451897e-07 1.3754805e-04 4.2866563e-05 6.8858487e-04]
    top_k:[4 9 7 8 6 5 3 2 1 0]

    #cpp
    not compiled


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
    cd /opt/intel/openvino/bin
    source setupvars.sh

    cd /opt/intel/openvino/deployment_tools/inference_engine/samples/
    bash build_samples.sh

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


# mnist_movidius_ncs
    bash convert_model.sh

    #python
    python3 hello_ncs.py
    python3 mnist_ncs.py

    #cpp
    bash build.sh
    ./mnist_ncs
    #or
    mkdir build&&cd build&&cmake ..&&make
    cp mnist ../
    cd ../&&mnist

    the results:
    Hello NCS! Device opened normally.
    Goodbye NCS! Device closed normally.
    NCS device working.
    
    Number of categories: 10
    Start download to NCS...
    *******************************************************************************
    mnist on NCS
    *******************************************************************************
    4 4 0.99902
    9 9 0.00069046
    7 7 0.00013757
    8 8 0.0
    6 6 0.0
    5 5 0.0
    3 3 0.0
    2 2 0.0
    1 1 0.0
    0 0 0.0
    *******************************************************************************


# mnist_libtorch
    #python
    python3 mnist_train_test.py.py

    #cpp
    cd cpp
    export Torch_DIR=/data/libtorch
    mkdir build&&cd build&&cmake ..&&make
    ./mnist

    the results:
    ok
    prediction: 4
    [ CPULongType{1} ]
    top10:  4  9  7  8  5  3  6  1  2  0
    [ CPULongType{1,10} ]
    4 9 7 8 5 3 6 1 2 0 


# onnxruntime
    python3 pytorch2onnx.py
    python3 mnist_onnx.py

    the results:
	[array([[8.0073503e-04, 2.8827257e-04, 1.9504252e-04, 1.3457091e-01,
        8.6817268e-04, 7.8707945e-01, 1.2973312e-03, 2.9687810e-04,
        1.4535046e-02, 6.0068104e-02]], dtype=float32)]



# mnist_cpp_http
    #mnist_crow_http_server
    #bin test
    cd mnist_cpp_http/mnist_crow_http_server/
    mkdir build&&cmake ..&&make
    ./mnist_bin

    #server
    cd mnist_cpp_http/mnist_crow_http_server/
    mkdir build&&cmake ..&&make
    ./mnist_server
    
    #client
    cd mnist_cpp_http/mnist_crow_http_server/
    python3 client.py

    the results:
    {'class': 4, 'out_image': 'iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAEU0lEQVRIDY3BjU8SfxgA8Oc57lDO6BZmuTQTX5an1UZTwHODfNkCnVZ/rV5TfMvLlNgSN1GnrURN5gs3QPGO+z6/7TY2mN78fT4IDhAR6hERACAi3EFEUAPBASLCHUSEiFCPiKAeggNEhP+NiKAGggNEBABE5DgOEZkNaiAibyMi0zQtyyIisCE44DiO53mPx9PU1MRxXLlcLhaLhmGQDRHdbnezrVKp5HI5XdcrlQrYEBwIguDz+To7O9vb2xExm83+/v378vKyUqkAgMvl8vl8/f39fr//6upqa2srm80ahgE2BAcej+f169fRaFSWZV3XNzY2ksnkv3//DMPgOM7r9fb29gaDwadPn+7u7q6vrx8fH5umCTYEB16vNxgMfv78ube3N5PJqKqaSqUuLi4YYx6P59WrV6FQqK+vL5/Pa5qWTqfz+TxjDGwIDiRJUhTly5cvz58/1zRtbm5uf3+/XC4LgtDa2jo4OBgOh91u9+bmpqZp2WzWMAyoQnAgSZKiKNPT048ePVpYWEgkEqenpwDw5MmTd+/eRSKRjo6Ow8PDRCKxvb1dKBQYY1CF4ECSJEVR4vE4z/Nfv3799u3b1dWVKIo9PT0fPnwIBAKlUmllZUXTtNPTU9M0iQiqEBxIkqQoSiwW4zhOVVVN00ql0osXLxRFGRsba2lpSafTqqqm02ld1xljUAPBgSRJw8PDU1NTDQ0N8/Pza2trlUrl7du3sVgsGAwahrG6uqqq6s7OTqFQIBtUITh4/Pjx0NDQzMxMa2trKpX6/v07IobD4fHx8ZcvX56cnCwsLMzPz2cymWKxSDaoQnAgiuLAwMDk5GQgECgUCgcHBy6Xq7+/v7u72zCMra2tpaWlHz9+ZLPZcrkMAEQEVQgOBEFoa2sLh8ORSMTv9zc0NAiCwPO8ruuZTGZzc/PXr19HR0fFYpExRkRQA8EBx3FNTU1+v394eDgajcqyjIgHBwfJZDKVSu3v7+dyuZubG8YYEUE9BAeIyPN8S0vL4ODgx48fZVm+uLjQNG19ff3w8DCfz5umSTa4A8EBIrrd7o6OjkgkEo1GRVFMJpMrKyt7e3u6rluWRURgIyKoh+DA5XJJkhQIBOLxuCzLf//+nZub29jYOD8/tyyLiKCKiKAewn0Q0ePxdHZ2jo2NjY6ONjY2rq2tqaq6u7t7fX1NRFCPiKAGwn0EQXj27FkoFIrFYm/evMnlcqqqJhKJP3/+3N7ewn2ICKoQ7iOKoizLU1NTExMTzc3NmUxmdnZ2eXn56Ojo9vYWqogIEcFGRFCFcAcier3e9+/ff/r0KRQKWZb18+fPxcXFVCp1dnZmmiYRQQ1EBAAigiqEOxBRFMW+vr6RkZGuri5d19O2k5OTUqnEGIOHINxHEASfz9fW1ub1ekul0tnZ2eXlZblcZowRETwE4T6IyPO8IAgul8uyLNM0LcsiIgAgIngIgjNEJCJEhBpEBA/5D/W5QDtd4WO0AAAAAElFTkSuQmCC', 'score': 0.9900000095367432}




    # mnist_libcurl_http_client
    #server
    cd mnist_flask/webpage2
    python3 server.py


    #client
    cd mnist_cpp_http/mnist_libcurl_http_client/
    mkdir build&&cmake ..&&make
    ./fd

    the results:
        result:{
            "predict_probability": "0.99738353", 
            "predict_result": "4", 
            "success": true
        }

# mnist_triton_inference_server
     #转化模型
     python3 convert2torchscript.py
     #模型部署
     docker pull nvcr.io/nvidia/tritonserver:20.03-py3
     docker run --gpus '"device=7"' --name="jxl-tritonserver" -d --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8020:8000 -p8021:8001 -p8022:8002 -v /home/jiangxiaolong/triton-inference-server:/models  nvcr.io/nvidia/tritonserver:20.03-py3 trtserver --model-repository=/models
     #模型测试
     python3 triton_client.py
     the results:
     [[(4, -0.05619096755981445, ''), (8, -4.068029880523682, ''), (1, -4.242851734161377, ''), (9, -4.946555137634277, ''), (6, -4.998610019683838, ''), (7, -5.519482135772705, ''), (5, -5.806419372558594, ''), (2, -6.507933139801025, ''), (0, -7.803701400756836, ''), (3, -7.842188835144043, '')]]


# 四边形检测test_rect
    python3 server.py
![](test_rect/result.png)


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
> 
[https://github.com/movidius/ncsdk](https://github.com/movidius/ncsdk)
> 
[https://github.com/movidius/ncappzoo](https://github.com/movidius/ncappzoo)
> 
[https://pytorch.org/tutorials/advanced/cpp_frontend.html](https://pytorch.org/tutorials/advanced/cpp_frontend.html)
> 
[https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/label_image](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/label_image)
