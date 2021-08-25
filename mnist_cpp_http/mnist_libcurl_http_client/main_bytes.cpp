//
//  main.cpp
//  TestMain
//
//  Created by Li Cheng on 2018/10/12.
//  Copyright © 2018年 Li Cheng. All rights reserved.
//

#include <iostream>
#include "FaceDetect.hpp"
#include <md5.h>
#include <json.hpp>
using json = nlohmann::json;




int main(int argc, const char * argv[]) {
 

    const char *url = "http://192.168.1.94:5000/predict";
    
    const char *filePath = "../4.jpg";
    

    /*
    //MD5 from filename
    printf("md5file: %s\n", md5file(filePath).c_str());
  
    //MD5 from opened file
    std::FILE* file = std::fopen(filePath, "rb");
    printf("md5file: %s\n", md5file(file).c_str());
    std::fclose(file);
    */

    string md5value = md5file(filePath);

    string result = "";
    string request_id = "007";
    map<const char *, const char *> params;
    params.insert(map<const char *, const char *>::value_type("md5", md5value.c_str()));
    params.insert(map<const char *, const char *>::value_type("image", filePath));
    params.insert(map<const char *, const char *>::value_type("request_id", request_id.c_str()));

    FaceDetectApi facedetectApi = FaceDetectApi();
    facedetectApi.detect(url, params, filePath, result);
    std::cout<<"result:"<< result<< std::endl;


    // parse explicitly
    auto j3 = json::parse(result);
    std::cout<<j3["predict_result"]<<std::endl;




    return 0;
}
