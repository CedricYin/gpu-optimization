cmake_minimum_required(VERSION 3.12)
project(elementwise LANGUAGES CUDA)

include_directories(${CMAKE_SOURCE_DIR}/../include)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(main main.cu)