//
//  segment.cpp
//  MNN
//
//  Created by MNN on 2019/07/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "ImageProcess.hpp"
#include "Interpreter.hpp"
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include "Expr.hpp"
#include "ExprCreator.hpp"
#include "AutoTime.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <iostream>
using namespace std;
using namespace cv;

using namespace MNN;
using namespace MNN::CV;
using namespace MNN::Express;

int main(int argc, const char* argv[]) {

    auto model_path = "../mnn/mnist.mnn";
    auto inputPatch = "../../train_test_mnist/MNIST/testimage/4/1.jpg";
	



	std::shared_ptr<Interpreter> net(Interpreter::createFromFile(model_path));
	ScheduleConfig config;
	config.numThread = 1;
	config.type = MNN_FORWARD_AUTO;
	auto session = net->createSession(config);
    auto input = net->getSessionInput(session, "input");
	std::vector<int> shape = input->shape();
	int input_H=28;
	int input_W=28;
	net->resizeTensor(input, shape);
	net->resizeSession(session);







	//Image Preprocessing
	cv::Mat raw_image = cv::imread(inputPatch);

	cv::cvtColor(raw_image, raw_image, cv::COLOR_BGR2RGBA);

	int ori_height = raw_image.rows;
	int ori_width = raw_image.cols;
	double ratio = std::min(1.0 * input_H / ori_height, 1.0 * input_W / ori_width);
    int resize_height = int(ori_height * ratio);
	int resize_width = int(ori_width * ratio);



	raw_image.convertTo(raw_image, CV_32FC4);
	raw_image = raw_image/255 - 0.5;




	cv::Mat resized_image;
	cv::resize(raw_image, resized_image, cv::Size(resize_width, resize_height), 0, 0, cv::INTER_LINEAR);
	resized_image.convertTo(resized_image, CV_32FC4);



	//copy to input tensor
	std::vector<int> dim{1, input_H, input_W, 4};
	auto nhwc_Tensor = MNN::Tensor::create<float>(dim, NULL, MNN::Tensor::TENSORFLOW);
	auto nhwc_data = nhwc_Tensor->host<float>();
	auto nhwc_size = nhwc_Tensor->size();
	::memcpy(nhwc_data, resized_image.data, nhwc_size);
	input->copyFromHostTensor(nhwc_Tensor);




	//interface
	net->runSession(session);



	//post process
	auto output = net->getSessionOutput(session, "output");
	
	float *rr = output->host<float>();
	for (int ii =0;ii< output->elementSize();ii++)
		cout<<rr[ii]<<" ";
	


    return 0;
}
