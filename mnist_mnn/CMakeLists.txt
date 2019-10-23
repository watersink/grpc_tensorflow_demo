# put output to build dir

cmake_minimum_required(VERSION 3.7)
project(mnist)


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
set(CMAKE_CXX_FLAGS "-std=c++0x -Wno-deprecated")


set(OPENCV_INCLUDE_DIR /data/opencv-3.3.1/include)
set(OPENCV_LIBRARY_DIR /data/opencv-3.3.1/build/lib)
set(OPENCV_LIBS -lopencv_core -lopencv_features2d -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_imgcodecs)

set(MNN_LIBS /data/MNN/build/libMNN.so )

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


include_directories(/data/MNN/3rd_party/imageHelper)
include_directories(/data/MNN/include)
include_directories(/data/MNN/express/include)
include_directories(/data/MNN/schema/current)
include_directories(/data/MNN/3rd_party/flatbuffers/include)


add_executable(mnist mnist.cpp)
target_link_libraries(mnist dl  ${OpenCV_LIBS} ${MNN_LIBS})
