cmake_minimum_required(VERSION 3.10)


project(MNIST)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
set(CMAKE_CXX_FLAGS "-std=c++0x -Wno-deprecated")
#set(CMAKE_CXX_STANDARD 11)
#option(NCNN_OPENMP "openmp support" OFF)
FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()



set(OPENCV_INCLUDE_DIR /data/opencv-3.3.1/include)
set(OPENCV_LIBRARY_DIR /data/opencv-3.3.1/build/lib)
set(OPENCV_LIBS -lopencv_core -lopencv_features2d -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_imgcodecs)

include_directories(/data/ncnn/src)
include_directories(/data/ncnn/build/src)
include_directories(/usr/include)

set(NCNN_LIB /data/ncnn/build/src/libncnn.a)
#LINK_LIBRARIES(uuid)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})




add_executable(mnist mnist.cpp)
target_link_libraries(mnist dl ${NCNN_LIB} ${OpenCV_LIBS})
