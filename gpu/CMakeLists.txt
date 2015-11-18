cmake_minimum_required(VERSION 2.8.9)

project(nn_core_gpu)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAG "-g -Wall")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

list(APPEND CUDA_NVCC_FLAGS --gpu-architecture sm_20)
CUDA_INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}")

# Fine tune the parameter
add_definitions(-DBLOCK_SIZE=4)

# Build static library
CUDA_ADD_LIBRARY(mvm_gpu mvm.cu)

# Test
CUDA_ADD_EXECUTABLE(mvm_gpu_test test.cu)

enable_testing()
add_test(NAME mvm_gpu COMMAND mvm_gpu_test)

