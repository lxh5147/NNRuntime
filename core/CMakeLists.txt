cmake_minimum_required(VERSION 2.8.9)

project(nn_core)

set(CMAKE_CXX_FLAGS                "-Wall -std=c++11 -static-libstdc++")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g -DDEBUG -DPERF")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

#unit test
add_executable(nn_runtime_core_test test.cpp)

#coverage test
add_executable(nn_runtime_core_test_coverage test.cpp)
set_target_properties(nn_runtime_core_test_coverage PROPERTIES COMPILE_FLAGS "-g -O0 --coverage")
set_target_properties(nn_runtime_core_test_coverage PROPERTIES LINK_FLAGS "-g -O0 --coverage")

#perf test
add_executable(nn_runtime_core_test_perf test.cpp)
set_target_properties(nn_runtime_core_test_perf PROPERTIES COMPILE_FLAGS "-O4 -mtune=native -g -pg")
set_target_properties(nn_runtime_core_test_perf PROPERTIES LINK_FLAGS "-O4 -mtune=native -g -pg")
set_target_properties(nn_runtime_core_test_perf PROPERTIES COMPILE_DEFINITIONS "PERF")

#run tests
enable_testing()
get_target_property(nn_runtime_core_test_app nn_runtime_core_test LOCATION)
add_test(NAME nn_runtime_core COMMAND ${nn_runtime_core_test_app})

#manually code coverage, cpu and memory profiling
# cd core
# ./profile.sh

if(GPU_ENABLED)
    target_link_libraries(nn_runtime_core_test mvm_gpu)
    target_link_libraries(nn_runtime_core_test_coverage mvm_gpu)
    target_link_libraries(nn_runtime_core_test_perf mvm_gpu)
endif()
