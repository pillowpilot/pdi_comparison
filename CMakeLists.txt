project(PDITPG)
cmake_minimum_required(VERSION 3.10)

#################################
# OpenCV

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

##################################
# GoogleTest

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

#################################
# Project targets
include_directories(include)

add_executable(Main src/main)
target_link_libraries(Main ${OpenCV_LIBS})

#################################
# Tests targets
add_executable(BHE3PLTest tests/bhe3pl_tests)
target_link_libraries(BHE3PLTest ${OpenCV_LIBS} ${GTEST_BOTH_LIBRARIES})
add_test(BHE3PLTest BHE3PLTest)
