cmake_minimum_required(VERSION 3.5)
project(cppnn)

set (CMAKE_CXX_STANDARD 14)

set(CXX_FLAGS “-Wl --undefined dynamic_lookup”)
set(CMAKE_CXX_FLAGS “${CXX_FLAGS}”)

find_package( OpenCV REQUIRED )
find_package(Boost COMPONENTS system filesystem REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")

find_package(TensorFlow REQUIRED)

set(PROJECT_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIRS})

add_executable(cppnn main.cpp)

include_directories(${PROJECT_INCLUDE_DIRS})
target_link_libraries(cppnn ${tensorflowlib} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TensorFlow_LIBRARIES})

