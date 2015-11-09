cmake_minimum_required(VERSION 2.8.9)
set(CMAKE_CXX_FLAG "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

set(CUDA_HOST_COMPILER /usr/bin/gcc)
find_package(CUDA)

CUDA_INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}")
set(CUDA_SRCS mvm.cu)

set(BUILD_SHARED_LIBS ON)
list(APPEND CUDA_NVCC_FLAGS --gpu-architecture sm_20)

set(CUDASONAME "mvm_gpu")
CUDA_ADD_LIBRARY(${CUDASONAME} ${CUDA_SRCS} SHARED)

project(cuda_mvm)
CUDA_ADD_EXECUTABLE(mvm_gpu_test test.cu)

enable_testing()
get_target_property(mvm_gpu_test_app mvm_gpu_test LOCATION)
add_test(NAME mvm_gpu COMMAND ${mvm_gpu_test_app})


