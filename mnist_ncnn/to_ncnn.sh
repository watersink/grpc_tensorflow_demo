python3 ckpt2pb.py
tensorflow2ncnn ./mnist.pb ./ncnn/mnist.param ./ncnn/mnist.bin
mkdir build&&cd build&&cmake ..&&make
./mnist
