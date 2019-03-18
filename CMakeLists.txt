cmake_minimum_required(VERSION 3.12)
project(cppnn)


add_executable(cppnn main.cpp)

find_library(tensorflowlibfwk libtensorflow_framework.so)
find_library(tensorflowlibcc libtensorflow_cc.so)
target_link_libraries(cppnn ${tensorflowlibfwk} ${tensorflowlibcc})

