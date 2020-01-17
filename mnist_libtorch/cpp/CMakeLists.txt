cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mnist)

set(Torch_DIR "/data/libtorch/")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Torch REQUIRED)


add_executable(mnist mnist_libtorch.cpp)
target_link_libraries(mnist ${TORCH_LIBRARIES} ${OpenCV_LIBS})
