//
// Created by root on 7/3/19.
//



#include "MNIST_CPP.h"


#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

int main(){


    string modelpath="../freeze_optimize_quantize/quantize_graph.pb";
    MNIST_CPP mnist(modelpath);
    cv::Mat image=cv::imread("../train_test_mnist/MNIST/testimage/5/1.jpg",1);

    mnist.process(image);



    /*
     * //simple test
    Session* session;
    Status status = NewSession(tensorflow::SessionOptions(), &session);
    if (!status.ok()) {
        cout << status.ToString() << "\n";
        return 1;
    }
    cout << "Session successfully created.\n";
    */

    return 0;
}
