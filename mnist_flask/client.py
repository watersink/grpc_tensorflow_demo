import requests
import flask

class MNIST_CLIENT(object):
    def __init__(self):
        self.headers = {}
        self.headers['Content-Type'] = 'application/x-www-form-urlencoded'
        self.urls = 'http://127.0.0.1'
        self.port = '5000'
        self.route = 'predict'
 
 
    def post_ocr(self,image_name):
        img_bytes = open(image_name, "rb").read()
        json_values = {'image':img_bytes}

        out_json = requests.post("%s:%s/%s" % (self.urls, self.port, self.route), files=json_values).json()
        print(out_json)        


if __name__=="__main__":
    mnist_client=MNIST_CLIENT()
    image_name = '../train_test_mnist/MNIST/testimage/5/1.jpg'
    mnist_client.post_ocr(image_name)