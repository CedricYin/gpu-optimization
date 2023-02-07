cmake_minimum_required(VERSION 3.12)
project(sgemm LANGUAGES CUDA)

file(GLOB CUDA_FILES *.cu)

foreach(SOURCE_FILE ${CUDA_FILES})
    string(REGEX REPLACE ".+/(.+)\\..*" "\\1" FILE_NAME ${SOURCE_FILE})
    add_executable(${FILE_NAME} ${SOURCE_FILE})
endforeach(SOURCE_FILE)