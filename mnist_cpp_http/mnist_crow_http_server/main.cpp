#include "mnist.h"

int main(int argc, const char* argv[]) {
    MNIST mnist;
    std::string image_path = "../4.jpg";
    cv::Mat image = cv::imread(image_path ,0);

    cv::Mat out_image;
    float score_pred;
    int class_pred;
    mnist.process(image, out_image, score_pred, class_pred);
    std::cout<<"score_pred:"<<score_pred<<" class_pred:"<<class_pred<<std::endl;


    return 0;
}
