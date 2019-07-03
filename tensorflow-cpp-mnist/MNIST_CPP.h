//
// Created by root on 7/2/19.
//

#ifndef TF_EXAMPLE_MNIST_CPP_H
#define TF_EXAMPLE_MNIST_CPP_H


#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/framework/tensor.h>

#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;



using tensorflow::Session;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::GraphDef;

class MNIST_CPP {


public:

    MNIST_CPP(string modelpath);
    ~MNIST_CPP();
    void process(cv::Mat im);
    tensorflow::Session *loadModel(const string &pbFile, const string &model_name);


private:
    const int input_width = 28;
    const int input_height = 28;


    tensorflow::Session *lenetSession;


};


#endif //TF_EXAMPLE_MNIST_CPP_H
