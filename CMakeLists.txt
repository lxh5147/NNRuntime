cmake_minimum_required(VERSION 2.8.9)

# Enable C++11
#set(CMAKE_CXX_FLAGS                "-Wall -std=c++11")
#set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g")
#set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
#set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
#set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

project(nn_runtime)
set(CMAKE_BUILD_TYPE Release)

#build shared lib
include_directories(include)
file(GLOB SOURCES "src/*.cpp")
add_library(nn_runtime SHARD ${SOURCES})

#build test 
set ( PROJECT_LINK_LIBS libnn_runtime.so )
link_directories(build)
add_executable(test main.cpp)
target_link_libraries(test ${PROJECT_LINK_LIBS})

install(TARGETS nn_runtime DESTINATION /usr/lib)
