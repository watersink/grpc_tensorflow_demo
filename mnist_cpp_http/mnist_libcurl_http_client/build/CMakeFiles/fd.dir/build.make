# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/build

# Include any dependencies generated for this target.
include CMakeFiles/fd.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fd.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fd.dir/flags.make

CMakeFiles/fd.dir/FaceDetect.cpp.o: CMakeFiles/fd.dir/flags.make
CMakeFiles/fd.dir/FaceDetect.cpp.o: ../FaceDetect.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fd.dir/FaceDetect.cpp.o"
	/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fd.dir/FaceDetect.cpp.o -c /home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/FaceDetect.cpp

CMakeFiles/fd.dir/FaceDetect.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fd.dir/FaceDetect.cpp.i"
	/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/FaceDetect.cpp > CMakeFiles/fd.dir/FaceDetect.cpp.i

CMakeFiles/fd.dir/FaceDetect.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fd.dir/FaceDetect.cpp.s"
	/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/FaceDetect.cpp -o CMakeFiles/fd.dir/FaceDetect.cpp.s

CMakeFiles/fd.dir/CurlPost.cpp.o: CMakeFiles/fd.dir/flags.make
CMakeFiles/fd.dir/CurlPost.cpp.o: ../CurlPost.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/fd.dir/CurlPost.cpp.o"
	/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fd.dir/CurlPost.cpp.o -c /home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/CurlPost.cpp

CMakeFiles/fd.dir/CurlPost.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fd.dir/CurlPost.cpp.i"
	/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/CurlPost.cpp > CMakeFiles/fd.dir/CurlPost.cpp.i

CMakeFiles/fd.dir/CurlPost.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fd.dir/CurlPost.cpp.s"
	/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/CurlPost.cpp -o CMakeFiles/fd.dir/CurlPost.cpp.s

CMakeFiles/fd.dir/main.cpp.o: CMakeFiles/fd.dir/flags.make
CMakeFiles/fd.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/fd.dir/main.cpp.o"
	/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fd.dir/main.cpp.o -c /home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/main.cpp

CMakeFiles/fd.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fd.dir/main.cpp.i"
	/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/main.cpp > CMakeFiles/fd.dir/main.cpp.i

CMakeFiles/fd.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fd.dir/main.cpp.s"
	/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/main.cpp -o CMakeFiles/fd.dir/main.cpp.s

# Object files for target fd
fd_OBJECTS = \
"CMakeFiles/fd.dir/FaceDetect.cpp.o" \
"CMakeFiles/fd.dir/CurlPost.cpp.o" \
"CMakeFiles/fd.dir/main.cpp.o"

# External object files for target fd
fd_EXTERNAL_OBJECTS =

fd: CMakeFiles/fd.dir/FaceDetect.cpp.o
fd: CMakeFiles/fd.dir/CurlPost.cpp.o
fd: CMakeFiles/fd.dir/main.cpp.o
fd: CMakeFiles/fd.dir/build.make
fd: ../libs/curl/libcurl.so.4
fd: CMakeFiles/fd.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable fd"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fd.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fd.dir/build: fd

.PHONY : CMakeFiles/fd.dir/build

CMakeFiles/fd.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fd.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fd.dir/clean

CMakeFiles/fd.dir/depend:
	cd /home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client /home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client /home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/build /home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/build /home/jiangxiaolong/grpc_tensorflow_demo/mnist_cpp_http/mnist_libcurl_http_client/build/CMakeFiles/fd.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fd.dir/depend

