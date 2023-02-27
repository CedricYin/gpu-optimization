cmake_minimum_required(VERSION 3.10)
project(elementwise LANGUAGES CUDA)

add_compile_options(-std=c++14)
add_compile_options(-arch=sm_86)

include_directories(${CMAKE_SOURCE_DIR}/../../include)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(main main.cu)