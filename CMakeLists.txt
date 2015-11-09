cmake_minimum_required(VERSION 2.8.9)

project(nn_runtime)

add_subdirectory(core)
add_subdirectory(tools)
add_subdirectory(jni)

find_package(CUDA)
if (CUDA_FOUND)
        #compile gpu version
        message("CUDA found, using device matrix vector multiplier!")
        add_subdirectory(gpu)
        include_directories(gpu)
        add_definitions(-D_MVM_GPU_)
endif()


