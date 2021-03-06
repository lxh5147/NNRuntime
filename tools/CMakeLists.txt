cmake_minimum_required(VERSION 2.8.9)

project(nn_tool)

# Enable C++11
set(CMAKE_CXX_FLAGS                "-Wall -std=c++11 -static-libstdc++")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g -DDEBUG -DPERF")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

# Define include directories
include_directories(../core)

#Defines test suites
enable_testing()

# Build the tool
add_executable(import_theano_model "tool_import_theano_model.cpp")

#test
add_executable(theano_model_importer_test "test.cpp")

get_target_property(theano_model_importer_app import_theano_model LOCATION)
get_target_property(theano_model_importer_test_app theano_model_importer_test LOCATION)
get_filename_component(test_folder ${theano_model_importer_test_app} DIRECTORY)

#unzip toy model
set_source_files_properties(unzip_toy_model PROPERTIES GENERATED true)
add_custom_target(unzip_toy_model
    COMMAND unzip toy_nbow.model.zip -d toy_nbow_model
    WORKING_DIRECTORY ${test_folder})

#import toy model
set_source_files_properties(import_toy_model PROPERTIES GENERATED true)
add_custom_target(import_toy_model
    COMMAND ${theano_model_importer_app} toy_nbow_model toy_nbow_model.bin
    WORKING_DIRECTORY ${test_folder})

add_dependencies(import_toy_model unzip_toy_model)

add_test(NAME theano_model_importer_test COMMAND ${theano_model_importer_test_app} ${test_folder}/toy_nbow_model.bin )

add_dependencies(theano_model_importer_test import_toy_model)



