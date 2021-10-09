import requests
import flask
import hashlib
import os
import time
import json

class MNIST_CLIENT(object):
    def __init__(self):
        self.urls = 'http://127.0.0.1'
        self.port = '5000'
        self.route = 'predict_asy'


    def post_ocr(self,image_name):
        out_results = {}

        img_bytes = open(image_name, "rb").read()
        json_values={}
        json_files = {'image':img_bytes}

        #异步post
        out_json = requests.post("%s:%s/%s" % (self.urls, self.port, self.route), data=json_values,files=json_files).json()
        print(out_json)        

        #异步get
        if out_json["errorCode"]==0:
            get_url = "{}:{}{}".format(self.urls, self.port,out_json["location"])
            print(get_url)
            while True:
                time.sleep(1)
                response = requests.get(get_url)
                response = response.content.decode("utf-8")
                response_json = json.loads(response)
                print("**", response_json)
                if response_json["status"]=="Task completed!" or response_json['state']=='FAILURE':
                    break


            if response_json["status"]=="Task completed!" and response_json["state"]=="SUCCESS":
                out_results = response_json
            else:
                out_results = {}
        return out_results



if __name__=="__main__":
    mnist_client=MNIST_CLIENT()
    image_name = '../train_test_mnist/MNIST/testimage/5/1.jpg'
    out_results = mnist_client.post_ocr(image_name)
    print(out_results)
