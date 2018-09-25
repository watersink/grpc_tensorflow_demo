import os,sys
import cv2
import numpy as np
import tensorflow as tf




os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class MNIST():
    def __init__(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float32, [None, 28, 28, 3], name='input')
            conv1 = tf.layers.conv2d(inputs=self.inputs, filters=64, kernel_size=(3, 3), padding="same", activation=None)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=(3, 3), padding="same", activation=None)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
 
            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 128])
            fc1 = tf.layers.dense(pool2_flat, 500, activation=tf.nn.relu)
            fc2 = tf.layers.dense(fc1, 10, activation=tf.nn.relu)
            y_out = tf.nn.softmax(fc2,name='output')
            self.result = tf.argmax(y_out, 1,name='prediction')
            self.probability = tf.reduce_max(y_out,name='probability')

            init = tf.global_variables_initializer()
            self.session = tf.Session(graph=self.graph)
            self.session.run(init)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
            saver.restore(self.session, cur_dir+"/../train_test_mnist/save/mnist.ckpt")





            # Export model
            # WARNING(break-tutorial-inline-code): The following code snippet is
            # in-lined in tutorials, please update tutorial documents accordingly
            # whenever code changes.
            builder = tf.saved_model.builder.SavedModelBuilder("mnist/0")

            # Build the signature_def_map.
            tensor_info_inputs = tf.saved_model.utils.build_tensor_info(self.inputs)
            tensor_info_result = tf.saved_model.utils.build_tensor_info(self.result)
            tensor_info_probability = tf.saved_model.utils.build_tensor_info(self.probability)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'image': tensor_info_inputs},
                    outputs={'result': tensor_info_result,'probability': tensor_info_probability},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(
                self.session, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_images':
                        prediction_signature,
                },
                main_op=tf.tables_initializer(),
                )

            builder.save()

            print('Done exporting!')



if __name__=="__main__":
    mnist=MNIST()
