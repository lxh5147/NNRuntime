cmake_minimum_required(VERSION 2.8.9)

set(CUDA_HOST_COMPILER /usr/bin/gcc)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAG "-g -Wall")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

list(APPEND CUDA_NVCC_FLAGS --gpu-architecture sm_20)

# Build static library
CUDA_ADD_LIBRARY(mvm_gpu mvm.cu)


# Test
CUDA_ADD_EXECUTABLE(mvm_gpu_test test.cu)

enable_testing()
get_target_property(mvm_gpu_test_app mvm_gpu_test LOCATION)
add_test(NAME mvm_gpu COMMAND ${mvm_gpu_test_app})

