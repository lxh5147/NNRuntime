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
include_directories(../core)
include_directories(${JNI_INCLUDE_DIRS})

swig_add_module(nn_runtime_jni java nn_runtime.i "nn_runtime.h" "nn_runtime.cpp")

enable_testing()



# Define test suites
file(GLOB java_src nn_runtime_jni/*.java test.java)
add_jar(nn_runtime_jni_test ${java_src} ENTRY_POINT test)
get_target_property(jar_file nn_runtime_jni_test JAR_FILE)
get_target_property(so_file nn_runtime_jni LOCATION)
get_filename_component(nn_runtime_jni_folder ${so_file} DIRECTORY)

#cpp test
add_executable(nn_runtime_test "nn_runtime.cpp" "test.cpp")
get_target_property(nn_runtime_test_app nn_runtime_test LOCATION)
add_test(NAME nn_runtime_test COMMAND ${nn_runtime_test_app} ${nn_runtime_jni_folder}/../core/model.bin)

#jni test
add_test(NAME nn_runtime_jni_test COMMAND ${Java_JAVA_EXECUTABLE} -Djava.library.path=${nn_runtime_jni_folder} -jar ${jar_file} ${nn_runtime_jni_folder}/../core/model.bin)
