cmake_minimum_required(VERSION 3.16)
project(piv_cpp)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(src)

# for OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(piv_lib ${OpenCV_LIBS})
