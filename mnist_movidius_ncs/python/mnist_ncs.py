#! /usr/bin/env python3
from mvnc import mvncapi as mvnc
import sys
import numpy
import cv2

graph_filename = '../mnist.graph'
image_filename = '../1.jpg'

#mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 2)
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No devices found')
    quit()

device = mvnc.Device(devices[0])
device.OpenDevice()

#Load graph
with open(graph_filename, mode='rb') as f:
    graphfile = f.read()

#Load preprocessing data
mean = 127.5
std = 1/255

#Load categories
categories = ['0','1','2','3','4','5','6','7','8','9']
print('Number of categories:', len(categories))

#Load image size
reqsize = 28

graph = device.AllocateGraph(graphfile)

img = cv2.imread(image_filename).astype(numpy.float32)
dx,dy,dz= img.shape
delta=float(abs(dy-dx))
if dx > dy: #crop the x dimension
    img=img[int(0.5*delta):dx-int(0.5*delta),0:dy]
else:
    img=img[0:dx,int(0.5*delta):dy-int(0.5*delta)]
img = cv2.resize(img, (reqsize, reqsize))

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

for i in range(3):
    img[:,:,i] = (img[:,:,i] - mean) * std

print('Start download to NCS...')
graph.LoadTensor(img.astype(numpy.float16), 'user object')
output, userobj = graph.GetResult()

top_inds = output.argsort()[::-1][:10]

print(''.join(['*' for i in range(79)]))
print('mnist on NCS')
print(''.join(['*' for i in range(79)]))
for i in range(10):
    print(top_inds[i], categories[top_inds[i]], output[top_inds[i]])

print(''.join(['*' for i in range(79)]))
graph.DeallocateGraph()
device.CloseDevice()
print('Finished')
