//
// Created by zhangyiyou on 6/28/21.
//
#include "crow.h"
#include "base64.h"
#include <exception>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <json.hpp>
#include <openssl/md5.h>
#include "mnist.h"

using json = nlohmann::json;
using namespace std;
using namespace cv;


int PORT = 5000;


string get_md5(const string &str)
{
    MD5_CTX ctx;
    unsigned char sumdata[MD5_DIGEST_LENGTH];
    char tmp[33];
    memset(tmp, 0x00, 33);

    MD5_Init(&ctx);
    MD5_Update(&ctx, str.data(), str.size());
    MD5_Final(sumdata, &ctx);

    for (int j = 0; j < MD5_DIGEST_LENGTH; j++) {
        sprintf(&tmp[(j * 2)], "%02x", (int) sumdata[j]);
    }

    string md5 = string(tmp);
    return md5;
}



std::string image_to_base64(cv::Mat img){
    std::vector<uchar> buf;
    cv::imencode(".png", img, buf);
    auto *enc_msg = reinterpret_cast<unsigned char*>(buf.data());
    std::string encoded = base64_encode(enc_msg, buf.size());
    return encoded;
}



cv::Mat base64_to_image(std::string encoded){
    string dec_jpg =  base64_decode(encoded);
    std::vector<uchar> data(dec_jpg.begin(), dec_jpg.end());
    cv::Mat img = cv::imdecode(cv::Mat(data), 1);
    return img;
}


int main() {
    MNIST *mnist = NULL;

    if (mnist == NULL) {
        mnist = new MNIST();
    }


    // App
    crow::SimpleApp app;
    CROW_ROUTE(app, "/predict").methods("POST"_method)
            ([mnist](const crow::request &req) {
                auto args = crow::json::load(req.body);
		if (!args)
                {
                    return crow::response(400);
                }


                std::string base64_image = args["image_base64"].s();
                std::string img_data = base64_decode(base64_image);
                std::string md5_value = get_md5(img_data);



                //std::vector<uchar> data(image_data.begin(), image_data.end());
                //cv::Mat image = cv::imdecode(cv::Mat(data), 1);

                cv::Mat image = base64_to_image(base64_image);
		std::cout<<image.cols<<"  "<<image.rows<<" "<<image.channels()<<std::endl;
		//cv::imwrite("test.jpg",image);




                /*
		* * file write for test
		std::string image_data = base64_decode(base64_image);
		string new_jpg = "test.jpg";
                std::ofstream fout;
                fout.open(new_jpg);
                fout << image_data << endl;
                fout.close();
		*/


                std::string md5 = args["md5"].s();
                std::string request_id = args["request_id"].s();
                int return_score = args["return_score"].i();

                std::cout<<"md5:"<<md5<<std::endl;
                std::cout<<"request_id:"<<request_id<<std::endl;
                std::cout<<"return_score:"<<return_score<<std::endl;




                json json_result;
		if (md5_value == md5){
	            try {
		        cv::Mat out_image;
                        float score_pred;
                        int class_pred;
                        mnist->process(image, out_image, score_pred, class_pred);
                        std::cout<<"score_pred:"<<score_pred<<" class_pred:"<<class_pred<<std::endl;


                        json_result["out_image"] = image_to_base64(out_image);
		        json_result["score"] = score_pred;
		        json_result["class"] = class_pred;

                    }
                    catch (std::exception &e) {
                        std::cout << "exception caught: " << e.what() << '\n';
                        json_result["errorCode"] = "-1";
	                json_result["errorMsg"] = std::string(e.what());

                    }
	
		}
	        else{
                    std::cout << "md5 is wrong\n";
                    json_result["errorCode"] = "-2";
	            json_result["errorMsg"] = "md5 is wrong";
		
		}

		std::string json_result_string = json_result.dump();



                return crow::response(json_result_string);

            });
	    


    app.bindaddr("0.0.0.0").port(PORT).multithreaded().run();
    return 0;
}
