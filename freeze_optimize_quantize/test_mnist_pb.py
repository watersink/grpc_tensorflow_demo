import os,sys
import cv2
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class MNIST():
    def __init__(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        #model_file = "./frozen_graph.pb"
        #model_file = "./optimize_graph.pb"
        model_file = "./quantize_graph.pb"
        self.graph = self.load_graph(model_file)
        with self.graph.as_default():
            self.inputs = self.graph.get_tensor_by_name("import/input:0")
            self.result = self.graph.get_tensor_by_name("import/prediction:0")
            self.probability = self.graph.get_tensor_by_name("import/probability:0")


            self.session = tf.Session(graph=self.graph)
    def load_graph(self, model_file):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        return graph
    def interface(self,input):
        array = np.array(input)/(255*1.0)-0.5
        samples_features =  array.reshape([-1,28,28,3])
        feed = {self.inputs: samples_features}
        predict_result,predict_probability = self.session.run([self.result,self.probability], feed_dict=feed)
        return predict_result,predict_probability

if __name__=="__main__":
    mnist=MNIST()
    image=cv2.imread("../train_test_mnist/MNIST/testimage/5/1.jpg")
    predict_result,predict_probability=mnist.interface(image)
    print(predict_result,predict_probability)
