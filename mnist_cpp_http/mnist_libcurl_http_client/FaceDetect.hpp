//
//  FaceDetect.hpp
//  FaceppApiLib
//
//  Created by Li Cheng on 2018/9/29.
//  Copyright © 2018年 Li Cheng. All rights reserved.
//

#ifndef FaceDetect_hpp
#define FaceDetect_hpp

#include <string>
#include <map>

using namespace std;

class FaceDetectApi {
public:
    void detect(const char *url, map<const char *, const char *> &params, const string &filePath, string &result);
};


#endif /* FaceDetect_hpp */

