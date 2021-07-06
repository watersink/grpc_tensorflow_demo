//
//  CurlPost.cpp
//  FaceppApiLib
//
//  Created by Li Cheng on 2018/9/30.
//  Copyright © 2018年 Li Cheng. All rights reserved.
//

#include "CurlPost.hpp"
#include <sstream>
#include <curl/curl.h>
#include <curl/easy.h>
#include <string.h>

using namespace std;

static size_t call_back(char *ptr, size_t size, size_t nmemb, void *content) {
    string data(ptr, size * nmemb);
    *((stringstream *) content) << data;

    return size * nmemb;
}

void CurlPost::doPost(const char *URL, map<const char *, const char *> params, string &result) {

    map<const char *, const char *>::iterator iter;

    struct curl_httppost *post = NULL;
    struct curl_httppost *last = NULL;

    // 初始化
    curl_global_init(CURL_GLOBAL_ALL);

    for (iter = params.begin(); iter != params.end(); iter++) {
        const char *key = iter->first;
        const char *value = iter->second;

        if (0 == strcasecmp("image", key)
            || 0 == strcasecmp("image1", key)
            || 0 == strcasecmp("image2", key)
            || 0 == strcasecmp("template_file", key)
            || 0 == strcasecmp("merge_file", key)) {
            curl_formadd(&post, &last,
                         CURLFORM_COPYNAME, key,
                         CURLFORM_FILE, value,
                         CURLFORM_END);
        } else {
            curl_formadd(&post, &last,
                         CURLFORM_COPYNAME, key,
                         CURLFORM_COPYCONTENTS, value,
                         CURLFORM_END);
        }
    }

    stringstream response;
    CURL *curl = curl_easy_init();
    curl_easy_setopt(curl, CURLOPT_URL, URL);
    curl_easy_setopt(curl, CURLOPT_HTTPPOST, post);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &call_back);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    // 执行
    CURLcode rescode = curl_easy_perform(curl);
    if (CURLE_OK != rescode) {
        fprintf(stderr, "\n\n-------请求失败-------\n %s \n\n", curl_easy_strerror(rescode));
    }

    result = response.str();

    //释放资源
    curl_easy_cleanup(curl);
    curl_formfree(post);
};

