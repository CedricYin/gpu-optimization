cmake_minimum_required(VERSION 3.12)
project(sgemm LANGUAGES CUDA)

add_compile_options(-std=c++14)
add_compile_options(-arch=sm_86)

file(GLOB CUDA_FILES *.cu)

include_directories(${CMAKE_SOURCE_DIR}/../../include)
include_directories(${CMAKE_SOURCE_DIR}/include)

foreach(SOURCE_FILE ${CUDA_FILES})
    string(REGEX REPLACE ".+/(.+)\\..*" "\\1" FILE_NAME ${SOURCE_FILE})
    add_executable(${FILE_NAME} ${SOURCE_FILE})
endforeach(SOURCE_FILE)