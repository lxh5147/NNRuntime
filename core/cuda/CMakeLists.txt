cmake_minimum_required(VERSION 2.8.9)
set(CMAKE_CXX_FLAG "-g -Wall")

set(CUDA_HOST_COMPILER /usr/bin/gcc)
find_package(CUDA)

CUDA_INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}")
set(CUDA_SRCS MatrixVectoryMultiplier.cu)

set(BUILD_SHARED_LIBS ON)
list(APPEND CUDA_NVCC_FLAGS --gpu-architecture sm_20)

set(CUDASONAME "mvm_cuda")
CUDA_ADD_LIBRARY(${CUDASONAME} ${CUDA_SRCS} SHARED)

project(cuda_mvm)
CUDA_ADD_EXECUTABLE(cuda_mvm_test test.cu)

enable_testing()
get_target_property(cuda_mvm_test_app cuda_mvm_test LOCATION)
add_test(NAME cuda_mvm COMMAND ${cuda_mvm_test_app})


