//
//  CurlPost.hpp
//  FaceppApiLib
//
//  Created by Li Cheng on 2018/9/30.
//  Copyright © 2018年 Li Cheng. All rights reserved.
//

#ifndef CurlPost_hpp
#define CurlPost_hpp

#include <iostream>
#include <map>
#include <string>

using namespace std;

using namespace std;

class CurlPost {
public :
    void doPost(const char *URL, map<const char *, const char *> params, string &result);
};

#endif /* CurlPost_hpp */

