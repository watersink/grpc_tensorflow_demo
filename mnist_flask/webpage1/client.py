import requests
import flask
import hashlib
import os

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
 
    def post_ocr(self,image_name):
        img_bytes = open(image_name, "rb").read()
        json_values={'md5':self.md5sum(image_name)}
        json_files = {'image':img_bytes}

        out_json = requests.post("%s:%s/%s" % (self.urls, self.port, self.route), data=json_values,files=json_files).json()
        print(out_json)        


if __name__=="__main__":
    mnist_client=MNIST_CLIENT()
    image_name = '../../train_test_mnist/MNIST/testimage/5/1.jpg'
    mnist_client.post_ocr(image_name)
