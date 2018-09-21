import requests
import base64

class MNIST_CLIENT(object):
    def __init__(self):
        self.headers = {}
        self.headers['Content-Type'] = 'application/x-www-form-urlencoded'
        self.urls = 'http://127.0.0.1'
        self.port = '8000'
        self.route = 'predict'

    def pic_base64(self,filename):
        byte_content = open(filename, 'rb').read()
        ls_f = base64.b64encode(byte_content)
        return ls_f

    def post_ocr(self,image_name):
        img_base64 = self.pic_base64(image_name).decode("utf-8")
        json_values = {'image':img_base64}

        self.headers['Content-Length']=str(len(json_values))
        out_json = requests.post("%s:%s/%s" % (self.urls, self.port, self.route),  data=json_values,headers=self.headers).json()
        print(out_json)        


if __name__=="__main__":
    mnist_client=MNIST_CLIENT()
    image_name = '../../train_test_mnist/MNIST/testimage/5/1.jpg'
    mnist_client.post_ocr(image_name)