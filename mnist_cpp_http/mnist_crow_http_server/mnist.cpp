#include "mnist.h"


MNIST::MNIST(){

    std::cout << "load model ok\n";
    
}



MNIST::~MNIST(){
}


int MNIST::process(cv::Mat image, cv::Mat &out_image, float &score_pred, int &class_pred){

    //gauss blure
    cv::GaussianBlur(image, out_image ,cv::Size(5,5),0);

    score_pred = 0.99;
    class_pred = 4;

    return 0;

}

