#!/usr/bin/env python
import pika
import uuid
import base64
import json



class MNIST_CLIENT(object):
    def __init__(self):
        self.host_ip='127.0.0.1'
        self.queue_name='mnist_queue'
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host_ip))

        self.channel = self.connection.channel()

        result = self.channel.queue_declare(exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(self.on_response, no_ack=True,
                                   queue=self.callback_queue)
    def pic_base64(self,filename):
        byte_content = open(filename, 'rb').read()
        ls_f = base64.b64encode(byte_content)
        return ls_f

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, image_name):
        img_base64 = self.pic_base64(image_name).decode("utf-8")
        json_values = {'image':img_base64}
        body_str=json.dumps(json_values, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(exchange='',
                                   routing_key=self.queue_name,
                                   properties=pika.BasicProperties(
                                         reply_to = self.callback_queue,
                                         correlation_id = self.corr_id,
                                         ),
                                   body=body_str)
        while self.response is None:
            self.connection.process_data_events()
        return self.response.decode("utf-8")


if __name__=='__main__':
    mnist_client = MNIST_CLIENT()
    image_name = '../train_test_mnist/MNIST/testimage/5/1.jpg'
    response = mnist_client.call(image_name)
    print(response)
