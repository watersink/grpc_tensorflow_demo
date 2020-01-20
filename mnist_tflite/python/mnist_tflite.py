# -*- coding:utf-8 -*-
import os
import numpy as np
import cv2
import tensorflow as tf


os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
model_path = "../mnist.tflite"

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
print(str(input_details))
output_details = interpreter.get_output_details()
print(str(output_details))





img =cv2.imread("../1.jpg",1)
img = cv2.resize(img,(28,28))
x = np.expand_dims(img, axis=0)
x = x.astype('float32')  # 类型也要满足要求
x = (x-127.5)/255.0

# 填装数据
interpreter.set_tensor(input_details[0]['index'], x)

# 注意注意，我要调用模型了
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# 出来的结果去掉没用的维度
results = np.squeeze(output_data)
print('results:{}'.format(results))
top_k = results.argsort()[:][::-1]
print("top_k:{}".format(top_k))
