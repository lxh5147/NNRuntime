cmake_minimum_required(VERSION 2.8.9)

# Enable C++11
set(CMAKE_CXX_FLAGS                "-Wall -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

#test
project(nn_core)
add_executable(nn_runtime_core_test test.cpp)

#make test to run test
enable_testing()
get_target_property(nn_runtime_core_test_app nn_runtime_core_test LOCATION)
add_test(NAME nn_runtime_core COMMAND ${nn_runtime_core_test_app})


