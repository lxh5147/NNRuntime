cmake_minimum_required(VERSION 2.8.9)

project(nn_runtime_jni)

set(CMAKE_CXX_FLAGS                "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -g -DDEBUG -DPERF")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

find_package(Java REQUIRED COMPONENTS Runtime Development)
#ensure that java home is properly set, e.g., JAVA_HOME=/usr/lib/jvm/java-8-oracle/
find_package(JNI REQUIRED)
find_package(SWIG REQUIRED)
include(UseJava)
include(UseSWIG)

#Boost 
find_package (Boost REQUIRED)
include_directories (SYSTEM ${Boost_INCLUDE_DIR})

#pthread
find_package (Threads)

set(CMAKE_SWIG_FLAGS -package nn_runtime_jni)
set(CMAKE_SWIG_OUTDIR "nn_runtime_jni")
SET_SOURCE_FILES_PROPERTIES(nn_runtime.i PROPERTIES CPLUSPLUS 1)

set(CMAKE_JNI_TARGET TRUE)

#include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${JNI_INCLUDE_DIRS})

swig_add_module(nn_runtime_jni java nn_runtime.i "nn_runtime.h" "nn_runtime.cpp")
target_link_libraries(nn_runtime_jni ${CMAKE_THREAD_LIBS_INIT})

enable_testing()

# Define test suites
file(GLOB java_src ${CMAKE_CURRENT_BINARY_DIR}/nn_runtime_jni/*.java test.java)
add_jar(nn_runtime_jni_test ${java_src} ENTRY_POINT test)
get_target_property(jar_file nn_runtime_jni_test JAR_FILE)


#cpp test
add_executable(nn_runtime_test "nn_runtime.cpp" "test.cpp")
add_test(NAME nn_runtime_test COMMAND nn_runtime_test )
target_link_libraries(nn_runtime_test ${CMAKE_THREAD_LIBS_INIT})

#jni test
#enable assert key words with -ea option
add_test(NAME nn_runtime_jni_test COMMAND ${Java_JAVA_EXECUTABLE} -ea -Djava.library.path=${CMAKE_CURRENT_BINARY_DIR} -jar ${jar_file} model.test.bin)

if(GPU_ENABLED)
        target_link_libraries(nn_runtime_jni mvm_gpu)
        target_link_libraries(nn_runtime_test mvm_gpu)
endif()

