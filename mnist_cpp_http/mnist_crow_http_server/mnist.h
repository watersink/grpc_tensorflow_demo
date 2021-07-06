#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <memory>
using namespace std;
using namespace cv;

class MNIST {
public:
    MNIST();
    ~MNIST();

    int process(cv::Mat image, cv::Mat &out_image, float &score_pred, int &class_pred);
};
