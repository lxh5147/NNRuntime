cmake_minimum_required(VERSION 2.8.9)

project(nn_core)

set(CMAKE_CXX_C11                  "-std=c++11 -static-libstdc++")
set(CMAKE_CXX_FLAGS                "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g -DDEBUG -DPERF")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

#unit test
add_executable(nn_runtime_core_test test.cpp)
set_target_properties(nn_runtime_core_test PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_C11}")

#perf test
add_executable(nn_runtime_core_perf_test test.cpp)
set_target_properties(nn_runtime_core_perf_test PROPERTIES COMPILE_FLAGS "-O4 -mtune=native -g -pg ${CMAKE_CXX_C11}")
set_target_properties(nn_runtime_core_perf_test PROPERTIES COMPILE_DEFINITIONS "PERF")

#run tests
enable_testing()
get_target_property(nn_runtime_core_test_app nn_runtime_core_test LOCATION)
add_test(NAME nn_runtime_core COMMAND ${nn_runtime_core_test_app})

#manually run performance test with ./nn_runtime_core_perf_test [perfReal|perf]
find_package(CUDA)
if (CUDA_FOUND)
        #compile gpu version
        message("CUDA found, using device matrix vector multiplier!")
        add_subdirectory(gpu)
        set_target_properties(nn_runtime_core_test nn_runtime_core_perf_test PROPERTIES COMPILE_DEFINITIONS "_MVM_GPU_")
        target_link_libraries(nn_runtime_core_test mvm_gpu)
        target_link_libraries(nn_runtime_core_perf_test mvm_gpu)
endif()
