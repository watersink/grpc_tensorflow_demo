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
            saver.restore(self.session, cur_dir+"/save/mnist.ckpt")
    def interface(self,input):
        array = np.array(input)/(255*1.0)-0.5
        samples_features =  array.reshape([-1,28,28,3])
        feed = {self.inputs: samples_features}
        predict_result,predict_probability = self.session.run([self.result,self.probability], feed_dict=feed)
        return predict_result,predict_probability

if __name__=="__main__":
    mnist=MNIST()
    image=cv2.imread("./MNIST/testimage/5/1.jpg")
    predict_result,predict_probability=mnist.interface(image)
    print(predict_result,predict_probability)
