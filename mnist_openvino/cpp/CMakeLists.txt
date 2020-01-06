cmake_minimum_required(VERSION 3.10)

set(TARGET_NAME "mnist")
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



set(OPENCV_INCLUDE_DIR /opt/intel/openvino/opencv/include)
set(OPENCV_LIBRARY_DIR /opt/intel/openvino/opencv/lib)
set(OPENCV_LIBS -lopencv_core -lopencv_features2d -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_imgcodecs)

include_directories(/opt/intel/openvino/inference_engine/include 
                    /opt/intel/openvino/deployment_tools/inference_engine/src/extension 
                    /opt/intel/openvino/deployment_tools/inference_engine/samples/common/format_reader 
                    /opt/intel/openvino/deployment_tools/inference_engine/samples/common)

set(OPENVINO_LIBRARY_DIR /opt/intel/openvino/inference_engine/lib/intel64 
                         /home/wangbo/inference_engine_samples_build/intel64/Release/lib
                         /opt/intel/openvino_2019.1.094/deployment_tools/inference_engine/external/tbb/lib)


link_directories(${OPENVINO_LIBRARY_DIR})
set(OPENVINO_LIBS -linference_engine -lcpu_extension -lformat_reader -ltbb)
#LINK_LIBRARIES(uuid)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(mnist  mnist.cpp)
target_link_libraries(mnist 
                      dl
                      gflags
                      pthread
                      cpu_extension
                      ${OPENVINO_LIBS} 
                      ${OpenCV_LIBS}
                      ${InferenceEngine_LIBRARIES})


add_executable(mnist_async  mnist_async.cpp)
target_link_libraries(mnist_async 
                      dl
                      gflags
                      pthread
                      cpu_extension
                      ${OPENVINO_LIBS} 
                      ${OpenCV_LIBS}
                      ${InferenceEngine_LIBRARIES})

