cmake_minimum_required(VERSION 3.16)
project(kuiper_course)

set(CMAKE_CXX_STANDARD 17)

# # 由于glog、gtest、armadillo都是通过apt安装，没有xx。cmake，因此无法find_package
# # 所以全部手动指定
set(GLOG_INCLUDE_DIRS /usr/include)
set(GLOG_LINK_DIRS /usr/lib/x86_64-linux-gnu)

set(GTEST_INCLUDE_DIRS /usr/include)
set(GTEST_LINK_DIRS /usr/lib)

set(ARMADILLO_INCLUDE_DIRS /usr/include)
set(ARMADILLO_LINK_DIRS /usr/lib/x86_64-linux-gnu)

include_directories(./include)

# find_package(glog REQUIRED)
set(link_lib glog pthread gtest)
set(link_math_lib armadillo blas lapack)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
add_executable(kuiper_course main.cpp)

target_include_directories(kuiper_course PUBLIC /usr/include/armadillo_bits)
target_include_directories(kuiper_course PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(kuiper_course ${link_lib} ${link_math_lib})

enable_testing()
add_subdirectory(test)