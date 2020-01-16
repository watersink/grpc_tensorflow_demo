cmake_minimum_required(VERSION 3.10)

set(TARGET_NAME "mnist_ncs")
project(MNIST)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
set(CMAKE_CXX_FLAGS "-std=c++0x -Wno-deprecated")
set(CMAKE_CXX_STANDARD 11)



set(OPENCV_INCLUDE_DIR /opt/intel/openvino/opencv/include)
set(OPENCV_LIBRARY_DIR /opt/intel/openvino/opencv/lib)
set(OPENCV_LIBS -lopencv_core -lopencv_features2d -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_imgcodecs)
set(NCS_LIBS -lmvnc)

#LINK_LIBRARIES(uuid)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(mnist  mnist_ncs.cpp stb_image.h stb_image_resize.h fp16.c fp16.h)
target_link_libraries(mnist 
                      ${NCS_LIBS}
                      ${OpenCV_LIBS})


