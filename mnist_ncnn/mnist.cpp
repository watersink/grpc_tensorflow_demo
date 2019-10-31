
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <iostream>
#include "net.h"


using namespace std;
using namespace cv;



int main(int argc, const char* argv[]) {


    ncnn::Net net;
    const char *param_path_char = "../ncnn/mnist.param";
    const char *bin_path_char = "../ncnn/mnist.bin";
    net.load_param(param_path_char);
    net.load_model(bin_path_char);


    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
	const float norm_vals[3] = {0.003922f, 0.003922f, 0.003922f};
    auto inputPatch = "../../train_test_mnist/MNIST/testimage/1/2.jpg";

	cv::Mat bgr = cv::imread(inputPatch);
	cv::Mat bgr_resized;
	cv::resize(bgr,bgr_resized ,cv::Size(28, 28));
    //ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, bgr.cols, bgr.rows);
    ncnn::Mat in = ncnn::Mat::from_pixels(bgr_resized.data, ncnn::Mat::PIXEL_BGR2RGB, bgr_resized.cols, bgr_resized.rows);

    in.substract_mean_normalize(mean_vals, norm_vals);


    //ncnn::Mat in;// input blob as above
    ncnn::Mat out;
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input("input", in);
    ex.extract("output", out);
    cout<<out.h<<" "<<out.w<<" "<<out.c<<endl;


	float* p = out.channel(0);
	for (int i =0; i<out.h*out.w*out.c;i++)
		cout<<p[i]<<" ";
	cout<<endl;

}



