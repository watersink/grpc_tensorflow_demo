../../bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=input_graph.pb --input_checkpoint=../train_test_mnist/save/mnist.ckpt --output_graph=./frozen_graph.pb --output_node_names=prediction,probability

