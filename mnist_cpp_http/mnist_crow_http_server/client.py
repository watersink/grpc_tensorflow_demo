import json
import hashlib
import os
import base64
import requests


def md5sum(filename):
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



url_get = "http://localhost:5000/predict"
image_name = "4.jpg"

with open(image_name, 'rb') as f:
    image_base64_3 = base64.b64encode(f.read()).decode('utf-8')

data = {"image_base64":image_base64_3,"md5":md5sum(image_name), "request_id":"007", "return_score":1}

r = requests.post(url_get, data=json.dumps(data)).json()
#r = requests.post(url_get, json = data )
print(r)


