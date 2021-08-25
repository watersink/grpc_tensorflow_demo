//
//  main.cpp
//  TestMain
//
//  Created by Li Cheng on 2018/10/12.
//  Copyright © 2018年 Li Cheng. All rights reserved.
//

#include <iostream>
#include "FaceDetect.hpp"
#include <json.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "base64.h"
#include <openssl/md5.h>
using json = nlohmann::json;


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
    std::vector<int> vecCompression_params;
    vecCompression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    vecCompression_params.push_back(90);
    cv::imencode(".jpg", img, buf, vecCompression_params);
    //cv::imencode(".png", img, buf);
    auto *enc_msg = reinterpret_cast<unsigned char*>(buf.data());
    std::string encoded = base64_encode(enc_msg, buf.size());
    return encoded;
}

int main(int argc, const char * argv[]) {
 

    const char *url = "http://192.168.1.94:5000/predict";
    const char *filePath = "../4.jpg";
    cv::Mat image = cv::imread(filePath);
    string base64_image = image_to_base64(image);
    std::string img_data = base64_decode(base64_image);
    string md5value = get_md5(img_data);
    std::cout<<md5value<<std::endl;
    

    string result = "";
    map<const char *, const char *> params;
    params.insert(map<const char *, const char *>::value_type("md5", md5value.c_str()));
    string request_id = "007";
    params.insert(map<const char *, const char *>::value_type("base64", base64_image.c_str()));
    params.insert(map<const char *, const char *>::value_type("request_id", request_id.c_str()));



    FaceDetectApi facedetectApi = FaceDetectApi();
    facedetectApi.detect(url, params, filePath, result);
    std::cout<<"result:"<< result<< std::endl;


    // parse explicitly
    auto j3 = json::parse(result);
    std::cout<<j3["predict_result"]<<std::endl;




    return 0;
}
