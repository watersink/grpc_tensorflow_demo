cmake_minimum_required(VERSION 3.10)


project(fd)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
set(CMAKE_CXX_FLAGS "-std=c++0x -Wno-deprecated")
FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()



set(CURL_LIBRARY_DIR include/md5/ include/json/single_include/nlohmann/ include/ /usr/local/include/)
set(CURL_LIBS /home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/libs/curl/libcurl.so.4)


include_directories(${CURL_LIBRARY_DIR})

add_executable(fd FaceDetect.hpp FaceDetect.cpp CurlPost.hpp CurlPost.cpp main.cpp)
target_link_libraries(fd ${CURL_LIBS})
