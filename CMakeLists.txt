cmake_minimum_required(VERSION 3.5)
project(cppnn)

set (CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )
find_package(Boost COMPONENTS system filesystem REQUIRED)



add_executable(cppnn main.cpp)

include_directories(${PROJECT_INCLUDE_DIRS})
target_link_libraries(cppnn ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TensorFlow_LIBRARIES} tensorflow_cc)

