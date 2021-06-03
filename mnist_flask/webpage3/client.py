import requests
import flask
import hashlib
import os
import base64
import json



class MNIST_CLIENT(object):
    def __init__(self):
        self.urls = 'http://127.0.0.1'
        self.port = '5000'
        self.route = 'predict'

    def md5sum(self,filename):
        if not os.path.isfile(filename):
            return
        myhash = hashlib.md5()
        f = open(filename, 'rb')
        while True:
            b = f.read(8096)
            if not b:
                break
            myhash.update(b)
        f.close()
        return myhash.hexdigest()
 
    def post_ocr_bin(self,image_name):
        img_bytes = open(image_name, "rb").read()
        json_values={'md5':self.md5sum(image_name),  "request_id":"007"}
        json_files = {'image':img_bytes}

        out_json = requests.post("%s:%s/%s" % (self.urls, self.port, self.route), data=json_values,files=json_files).json()
        print("------post bin------\n", out_json) 


    def post_ocr_base64(self,image_name):
        with open(image_name, 'rb') as f:
            image_base64_3 = base64.b64encode(f.read())

        json_values={"md5":self.md5sum(image_name),  "request_id":"007", "base64": image_base64_3}
        out_json = json.dumps(requests.post("%s:%s/%s" % (self.urls, self.port, self.route), data=json_values).json()).encode().decode()
        print("------post base64------\n",out_json)       


    def post_ocr_url(self,image_url):
        response = requests.get(image_url)
        img_bytes = response.content

        cal_md5sum = hashlib.md5(img_bytes).hexdigest()
        json_values={'md5':cal_md5sum, "request_id":"007",'url':image_url}

        out_json = json.dumps(requests.post("%s:%s/%s" % (self.urls, self.port, self.route), data=json_values).json()).encode().decode()
        print("------post url------\n",out_json)




    def post_bash(self):
        #message_cmd = "cat /data/py_project/project/gunicorn_config.py"
        message_cmd = "ps -ef"
        #message_cmd = "free -g"
        #message_cmd = "top -n 1 -b "
        #message_cmd = "nvidia-smi"
        response = requests.post(self.urls+"/get_message", data={"message_cmd": message_cmd})
        for line in response.text.splitlines():
            print(line)




if __name__=="__main__":
    face_client= MNIST_CLIENT()
    image_name = '../../train_test_mnist/MNIST/testimage/5/1.jpg'
    image_url = "https://eli.thegreenplace.net/images/2016/mnist-test-9740.png"
    face_client.post_ocr_bin(image_name)
    face_client.post_ocr_base64(image_name)
    face_client.post_ocr_url(image_url)
    #face_client.post_bash()
