cmake_minimum_required(VERSION 2.8)
project (mnist)
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

if (MSVC)
	set(Boost_USE_STATIC_LIBS "On")
	find_package( Boost 1.52 COMPONENTS system thread regex REQUIRED )
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++1y -pedantic -Wextra")
	find_package( Boost 1.52 COMPONENTS system thread REQUIRED )
endif()



set(CROW_LIBRARY_DIR include/crow/include/ include/json/single_include/nlohmann/ include/ include/cpp-base64 /usr/local/include/ /usr/include )
set(OPENCV_INCLUDE_DIR /usr/local/include)
set(OPENCV_LIBRARY_DIR /usr/local/lib)
set(OPENCV_LIBS -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs)
set(Boost_LIBRARIES /usr/include/boost)
SET(Boost_USE_MULTITHREADED ON)


include_directories(${CROW_LIBRARY_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(mnist_server server.cpp mnist.cpp mnist.h include/cpp-base64/base64.h include/cpp-base64/base64.cpp)
target_link_libraries(mnist_server ${Boost_LIBRARIES} /usr/lib/ boost_thread boost_system)
target_link_libraries(mnist_server ${CMAKE_THREAD_LIBS_INIT} ${OPENCV_LIBS} ${TORCH_LIBRARIES} -lcrypto -lssl)


add_executable(mnist_bin mnist.cpp mnist.h main.cpp)
target_link_libraries(mnist_bin ${OPENCV_LIBS})
