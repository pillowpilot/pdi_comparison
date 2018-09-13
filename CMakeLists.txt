project(PDITPG)
cmake_minimum_required(VERSION 3.2)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(include)

add_executable(Main src/main)
target_link_libraries(Main ${OpenCV_LIBS})
