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



#test
add_executable(theano_model_importer_test "test.cpp")

get_target_property(test_exe_file theano_model_importer_test LOCATION)
get_filename_component(test_folder ${test_exe_file} DIRECTORY)

#unzip toy model
set_source_files_properties(unzip_toy_model PROPERTIES GENERATED true)
add_custom_target(unzip_toy_model
    COMMAND unzip toy_nbow.model.zip -d toy_nbow_model
    WORKING_DIRECTORY ${test_folder})
    
    
add_test(NAME theano_model_importer_test COMMAND ${theano_model_importer_test_app} ${test_folder}/toy_nbow_model )

add_dependencies(theano_model_importer_test unzip_toy_model)

# Build the tool
add_executable(import_theano_model "tool_import_theano_model.cpp")


