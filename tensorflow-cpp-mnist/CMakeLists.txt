cmake_minimum_required(VERSION 3.6)
project(mnist_cpp)

set(CMAKE_CXX_STANDARD 11)

SET(TF_DIR /data/tensorflow /data/tensorflow/bazel-genfiles /data/tensorflow/tensorflow/contrib/makefile/downloads/absl)
SET(TF_LIB /data/tensorflow/bazel-bin)
SET(Eigen_DIR /data/tensorflow/tensorflow/contrib/makefile/downloads/eigen)
SET(Protobuf_DIR /data/tensorflow/tensorflow/contrib/makefile/downloads/protobuf/src) # need make


INCLUDE_DIRECTORIES(${TF_DIR} ${OpenCV_INCLUDE_DIR} ${Eigen_DIR} ${Protobuf_DIR})
LINK_DIRECTORIES(${TF_LIB} ${Protobuf_DIR})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# 配置OpenMP
FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
include_directories(SYSTEM ${OpenMP_INCLUDE_PATH} /usr/local)
set(THIRED_LIB
        #        ${OpenMP_LIBS}
        ${OpenCV_LIBS}
        ${NCNN_Libs}
		${TF_LIB}
		/data/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so
        /data/tensorflow/bazel-bin/tensorflow/libtensorflow.so
        /data/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so
        )


add_executable(main  MNIST_CPP.h MNIST_CPP.cpp main.cpp)
target_link_libraries(main ${THIRED_LIB})

