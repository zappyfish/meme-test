cmake_minimum_required(VERSION 3.5)
project(cppnn)

set (CMAKE_CXX_STANDARD 11)

set (CMAKE_CXX_FLAGS "-Wl --allow-multiple-definition -Wl --whole-archive")

find_package( OpenCV REQUIRED )
find_package(Boost COMPONENTS system filesystem REQUIRED)

add_executable(cppnn main.cpp)

find_library(tensorflowlibfwk libtensorflow_framework.so)
find_library(tensorflowlibcc libtensorflow_cc.so)
target_link_libraries(cppnn ${tensorflowlibfwk} ${tensorflowlibcc} ${OpenCV_LIBS} ${Boost_LIBRARIES})

