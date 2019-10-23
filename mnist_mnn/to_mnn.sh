python3 ckpt2pb.py
/data/MNN/tools/converter/build/MNNConvert -f TF --modelFile mnist.pb --MNNModel mnn/mnist.mnn --bizCode MNN
/data/MNN/tools/converter/build/MNNDump2Json.out mnn/mnist.mnn mnist.json
/data/MNN/build/MNNV2Basic.out mnn/mnist.mnn 1 1
/data/MNN_bak/build/benchmark.out mnn 100 0
mkdir build&&cd build&&cmake ..&&make
./mnist
