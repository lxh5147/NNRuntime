cmake_minimum_required(VERSION 2.8.9)

project(nn_runtime_jni)

# Enable C++11
set(CMAKE_CXX_FLAGS                "-Wall -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

find_package(Java REQUIRED COMPONENTS Runtime Development)
#ensure that java home is properly set, e.g., JAVA_HOME=/usr/lib/jvm/java-8-oracle/
find_package(JNI REQUIRED)
find_package(SWIG REQUIRED)
include(UseJava)
include(UseSWIG)

set(CMAKE_SWIG_FLAGS -package nn_runtime_jni)
set(CMAKE_SWIG_OUTDIR "nn_runtime_jni")
SET_SOURCE_FILES_PROPERTIES(nn_runtime.i PROPERTIES CPLUSPLUS 1)

set(CMAKE_JNI_TARGET TRUE)

#include directories
include_directories(../include)
include_directories(${JNI_INCLUDE_DIRS})

swig_add_module(nn_runtime_jni java nn_runtime.i "nn_runtime.h" "nn_runtime.cpp")

enable_testing()

#cpp runtime test
add_executable(nn_runtime_test "nn_runtime.cpp" "test.cpp")
add_test(NAME nn_runtime_test COMMAND ${CMAKE_HOME_DIRECTORY}/nn_runtime_test)

#jni test
file(GLOB java_src nn_runtime_jni/*.java test.java)
add_jar(nn_runtime_jni_test ${java_src} ENTRY_POINT test)
get_target_property(_jarFile nn_runtime_jni_test JAR_FILE)
add_test(NAME nn_runtime_jni_test COMMAND ${Java_JAVA_EXECUTABLE} -Djava.library.path=${CMAKE_HOME_DIRECTORY} -jar ${_jarFile})
