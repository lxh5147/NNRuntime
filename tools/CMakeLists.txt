cmake_minimum_required(VERSION 2.8.9)

project(nn_tool)

set(CMAKE_CXX_FLAGS                "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g -DDEBUG -DPERF")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

#Boost 
find_package (Boost REQUIRED COMPONENTS regex)
include_directories (SYSTEM ${Boost_INCLUDE_DIR})

#pthread
find_package (Threads)

# Define include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../core)

#Defines test suites
enable_testing()

# Build the tool
add_executable(import_theano_model "tool_import_theano_model.cpp")
target_link_libraries(import_theano_model ${Boost_LIBRARIES})
target_link_libraries(import_theano_model ${CMAKE_THREAD_LIBS_INIT})

#test
add_executable(theano_model_importer_test "test.cpp")
target_link_libraries(theano_model_importer_test ${Boost_LIBRARIES})
target_link_libraries(theano_model_importer_test ${CMAKE_THREAD_LIBS_INIT})

#unzip toy model
set_source_files_properties(unzip_toy_model PROPERTIES GENERATED true)
add_custom_target(unzip_toy_model
    COMMAND unzip ${CMAKE_CURRENT_SOURCE_DIR}/toy_nbow.model.zip -d ${CMAKE_CURRENT_BINARY_DIR}/toy_nbow_model
)

#import toy model
set_source_files_properties(import_toy_model PROPERTIES GENERATED true)
add_custom_target(import_toy_model
    COMMAND import_theano_model ${CMAKE_CURRENT_BINARY_DIR}/toy_nbow_model ${CMAKE_CURRENT_BINARY_DIR}/toy_nbow_model.bin
)

add_dependencies(import_toy_model unzip_toy_model)

add_test(NAME theano_model_importer_test COMMAND theano_model_importer_test ${CMAKE_CURRENT_BINARY_DIR}/toy_nbow_model.bin )

add_dependencies(theano_model_importer_test import_toy_model)



