project(nn_runtime)
cmake_minimum_required(VERSION 2.8.9)

add_subdirectory(core)
add_subdirectory(tools)
add_subdirectory(jni)

#make test to run test
enable_testing()

