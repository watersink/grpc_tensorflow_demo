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

