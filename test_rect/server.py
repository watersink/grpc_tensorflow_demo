# coding=utf-8

import flask
from flask import request , render_template
import os,sys
import io
from PIL import Image
import numpy as np
import hashlib
import time
import base64
import uuid
import cv2
import json
import functools




app = flask.Flask(__name__)
@app.route('/testrect', methods=['GET'])
def index():
    return render_template('index.html')



#支持，自动归档，多进程，进程守护
if __name__ == '__main__':
    import logging
    app.logger.setLevel(logging.DEBUG)
    app.run(host="0.0.0.0", port=5000)

