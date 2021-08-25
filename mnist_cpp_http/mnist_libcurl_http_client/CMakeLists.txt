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


set(OPENCV_INCLUDE_DIR /usr/local/include)
set(OPENCV_LIBRARY_DIR /usr/local/lib64/)
set(OPENCV_LIBS -lopencv_core  -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs)

set(MD5_LIBRARY_DIR include/md5/ include/json/single_include/nlohmann/ include/ /usr/local/include/)
set(CURL_LIBRARY_DIR include/json/single_include/nlohmann/ include/ /usr/local/include/ include/cpp-base64/)
set(CURL_LIBS /home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/libs/curl/libcurl.so.4)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CURL_LIBRARY_DIR})

add_executable(fd_base64 FaceDetect.hpp FaceDetect.cpp CurlPost.hpp CurlPost.cpp main_base64.cpp include/cpp-base64/base64.cpp)
target_link_libraries(fd_base64 ${CURL_LIBS} ${OPENCV_LIBS}  /usr/local/lib64/libopencv_core.so.3.4.10 /usr/local/lib64/libopencv_imgcodecs.so.3.4.10 -lcrypto -lssl)
#target_link_libraries(fd ${CURL_LIBS} ${OPENCV_LIBS}  -lcrypto -lssl /usr/local/lib64/libopencv_core.so.3.4.10)

include_directories(${MD5_LIBRARY_DIR})
add_executable(fd_bytes FaceDetect.hpp FaceDetect.cpp CurlPost.hpp CurlPost.cpp main_bytes.cpp)
target_link_libraries(fd_bytes ${CURL_LIBS})
