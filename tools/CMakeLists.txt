cmake_minimum_required(VERSION 2.8.9)

project(nn_tool)

# Enable C++11
set(CMAKE_CXX_FLAGS                "-Wall -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

# Define include directories
include_directories(../core)

#Defines test suites
enable_testing()
add_executable(theano_model_importer_test "test.cpp")
get_target_property(theano_model_importer_test_app theano_model_importer_test LOCATION)
add_test(NAME theano_model_importer_test COMMAND ${theano_model_importer_test_app})

# Build the tool
add_executable(import_theano_model "tool_import_theano_model.cpp")


