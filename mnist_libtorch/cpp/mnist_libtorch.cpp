#include <torch/script.h> // One-stop header.
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>


std::string image_path = "../1.jpg";

int main(int argc, const char* argv[]) {

    // Deserialize the ScriptModule from a file using torch::jit::load().
	torch::jit::script::Module module = torch::jit::load("../mnist.pt");

    std::cout << "ok\n";

    //输入图像
	cv::Mat image = cv::imread(image_path ,0);
    cv::Mat image_transfomed;
    cv::resize(image, image_transfomed, cv::Size(28, 28));

    // 转换为Tensor
    torch::Tensor tensor_image = torch::from_blob(image_transfomed.data,
                            {image_transfomed.rows, image_transfomed.cols,1},torch::kByte);
    tensor_image = tensor_image.permute({2,0,1});
    tensor_image = tensor_image.toType(torch::kFloat);
    tensor_image = tensor_image.div(255);
    tensor_image = tensor_image.unsqueeze(0);

    // 网络前向计算
    at::Tensor output = module.forward({tensor_image}).toTensor();
	//std::cout << "output:" << output << std::endl;

	auto prediction = output.argmax(1);
	std::cout << "prediction:" << prediction << std::endl;

	int maxk = 10;
	auto top10 = std::get<1>(output.topk(maxk, 1, true, true));

	std::cout << "top10: " << top10 << '\n';

	std::vector<int> res;
	for (auto i = 0; i < maxk; i++) {
		res.push_back(top10[0][i].item().toInt());
	}
	for (auto i : res) {
		std::cout << i << " ";
	}
	std::cout << "\n";

}
