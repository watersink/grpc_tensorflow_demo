//
//  FaceDetect.cpp
//  FaceppApiLib
//
//  Created by Li Cheng on 2018/9/29.
//  Copyright © 2018年 Li Cheng. All rights reserved.
//

#include "FaceDetect.hpp"
#include <iostream>
#include <string>
#include <curl/curl.h>
#include "CurlPost.hpp"

using namespace std;

void FaceDetectApi::detect(const char *url, map<const char *, const char *> &params, const string &filePath, string &result) {

    if (filePath.empty()) {
        fprintf(stderr, "\n\n-------请求失败-------\n %s \n\n", "file path can not be empty !");
        return;
    }


    params.insert(map<const char *, const char *>::value_type("image", filePath.c_str()));


    CurlPost curlPost = CurlPost();
    curlPost.doPost(url, params, result);
}


