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

add_executable(AMBETest tests/ambe_tests)
target_link_libraries(AMBETest ${OpenCV_LIBS} ${GTEST_BOTH_LIBRARIES})
add_test(AMBETest AMBETest)

add_executable(PSNRTest tests/psnr_tests)
target_link_libraries(PSNRTest ${OpenCV_LIBS} ${GTEST_BOTH_LIBRARIES})
add_test(PSNRTest PSNRTest)

add_executable(EntropyTest tests/entropy_tests)
target_link_libraries(EntropyTest ${OpenCV_LIBS} ${GTEST_BOTH_LIBRARIES})
add_test(EntropyTest EntropyTest)

add_executable(ContrastTest tests/contrast_tests)
target_link_libraries(ContrastTest ${OpenCV_LIBS} ${GTEST_BOTH_LIBRARIES})
add_test(ContrastTest ContrastTest)
