find_package(GTest REQUIRED)

include_directories(../include/)

set(GLOG_INCLUDE_DIRS /usr/include)
set(GLOG_LINK_DIRS /usr/lib/x86_64-linux-gnu)

set(GTEST_INCLUDE_DIRS /usr/include)
set(GTEST_LINK_DIRS /usr/lib)

set(ARMADILLO_INCLUDE_DIRS /usr/include)
set(ARMADILLO_LINK_DIRS /usr/lib/x86_64-linux-gnu)

aux_source_directory(../test DIR_TEST)
set(link_lib glog gtest pthread)
set(link_math_lib armadillo blas lapack)

aux_source_directory(../source/data DIR_DATA)
add_executable(test_kuiper_course ${DIR_TEST} ${DIR_DATA})

link_directories(/usr/local/lib/)
target_link_libraries(test_kuiper_course ${link_lib} ${link_math_lib})
