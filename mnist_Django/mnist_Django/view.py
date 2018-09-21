from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import os,sys
import cv2
import numpy as np
import base64
import json

sys.path.append("../train_test_mnist/")
from test_mnist import MNIST

mnist=MNIST()

@csrf_exempt 
def MNIST_SERVER(request):
    if request.method=='GET':
        return HttpResponse("mnist django demo")
    elif request.method == 'POST':
        data = {"success": False}

        image_get = request.POST['image']
        image_file_value = base64.urlsafe_b64decode(image_get)
        image = cv2.imdecode(np.frombuffer(image_file_value, dtype='uint8'), 1)
        predict_result,predict_probability = mnist.interface(image)
        print("{}{}".format(predict_result,predict_probability))
        data["predict_result"] = str(predict_result[0])
        data["predict_probability"] = str(predict_probability)
        data["success"] = True
        return_data=json.dumps(data, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)
    return HttpResponse(return_data)