#!/usr/bin/env python
import pika
import base64
import cv2
import os,sys
import json
import numpy as np
sys.path.append("../train_test_mnist/")
from test_mnist import MNIST


mnist=MNIST()


class MNIST_SERVER(object):
    def __init__(self):
        self.host_ip='127.0.0.1'
        self.queue_name='mnist_queue'
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host_ip))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue_name)

    def on_request(self,ch, method, props, body):
        image_base64=json.loads(body.decode("utf-8"))['image']
        data = {"success": False}
        image_file_value = base64.urlsafe_b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(image_file_value, dtype='uint8'), 1)
        predict_result,predict_probability = mnist.interface(image)
        print("{}{}".format(predict_result,predict_probability))
        data["predict_result"] = str(predict_result[0])
        data["predict_probability"] = str(predict_probability)
        data["success"] = True
        
        response_data=json.dumps(data, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)



    
        ch.basic_publish(exchange='',
                         routing_key=props.reply_to,
                         properties=pika.BasicProperties(correlation_id = \
                                                             props.correlation_id),
                         body=response_data)
        ch.basic_ack(delivery_tag = method.delivery_tag)
    def on_consume(self):
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(self.on_request, queue=self.queue_name)
        
        print(" [x] Awaiting RPC requests")
        self.channel.start_consuming()

if __name__=="__main__":
    mnist_server=MNIST_SERVER()
    mnist_server.on_consume()
