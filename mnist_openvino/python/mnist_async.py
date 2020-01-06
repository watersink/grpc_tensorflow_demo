#!/usr/bin/env python
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IEPlugin


def main():
    model_xml = "./mnist.xml"
    model_bin = "./mnist.bin"
    input ="./1.jpg"
    number_iter=10
    device='CPU'
    perf_counts=True
    number_top =10
    batch_size=1



    # Plugin initialization for specified device and load extensions library if specified
    plugin = IEPlugin(device=device, plugin_dirs=None)
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    if plugin.device == device:
        supported_layers = plugin.get_supported_layers(net)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = batch_size

    # Read and pre-process input images
    n, c, h, w = net.inputs[input_blob].shape
    images = np.ndarray(shape=(n, c, h, w))
    for i in range(n):
        image = cv2.imread(input)
        image = image/255-0.5
        if image.shape[:-1] != (h, w):
            log.warning("Image {} is resized from {} to {}".format(input, image.shape[:-1], (h, w)))
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images[i] = image
    log.info("Batch size is {}".format(n))

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = plugin.load(network=net)

    # Start sync inference
    log.info("Starting inference ({} iterations)".format(number_iter))
    infer_time = []
    for i in range(number_iter):
        t0 = time()
        infer_request_handle = exec_net.start_async(request_id=0, inputs={input_blob: images})
        infer_request_handle.wait()
        infer_time.append((time() - t0) * 1000)
    log.info("Average running time of one iteration: {} ms".format(np.average(np.asarray(infer_time))))
    if perf_counts:
        perf_counts = infer_request_handle.get_perf_counts()
        log.info("Performance counters:")
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type', 'exet_type', 'status', 'real_time, us'))
        for layer, stats in perf_counts.items():
            print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer, stats['layer_type'], stats['exec_type'],
                                                              stats['status'], stats['real_time']))
    # Processing output blob
    log.info("Processing output blob")
    res = infer_request_handle.outputs[out_blob]
    log.info("Top {} results: ".format(number_top))

    classid_str = "classid"
    probability_str = "probability"
    for i, probs in enumerate(res):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)[-number_top:][::-1]
        print("Image {}\n".format(input))
        print(classid_str, probability_str)
        print("{} {}".format('-' * len(classid_str), '-' * len(probability_str)))
        for id in top_ind:
            space_num_before_prob = (len(probability_str) - len(str(probs[id]))) // 2
            print("{}         {:.7f}".format(id, probs[id]))
        print("\n")


if __name__ == '__main__':
    sys.exit(main() or 0)
