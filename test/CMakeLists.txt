cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb -fopenmp -mavx2")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -fopenmp -mavx2")
SET(CMAKE_ALL_ERRORS_FATAL TRUE)

project(TESTUNITS)

# find_package(Arrow REQUIRED)
# find_package(Parquet REQUIRED)
# find_package(edlib CONFIG REQUIRED)

set(EXECUTABLE_NAME "set" CACHE STRING "Choose which executable to build (set, overlap, str)")
set(COMMON_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common")

if(EXECUTABLE_NAME STREQUAL "set")
    add_executable(set test_setjoin.cc 
                       ${COMMON_INCLUDE}/io.cc 
                       ${COMMON_INCLUDE}/dataframe.cc 
                       ${COMMON_INCLUDE}/tokenizer.cc 
                       ${COMMON_INCLUDE}/setjoin.cc 
                       ${COMMON_INCLUDE}/setjoin_parallel.cc 
                       ${COMMON_INCLUDE}/joinutil.cc
                       ${COMMON_INCLUDE}/simfunc.cc)
    target_include_directories(set PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../cpp)
elseif(EXECUTABLE_NAME STREQUAL "overlap")
    add_executable(overlap test_ovlpjoin.cc 
                           ${COMMON_INCLUDE}/io.cc 
                           ${COMMON_INCLUDE}/dataframe.cc
                           ${COMMON_INCLUDE}/tokenizer.cc
                           ${COMMON_INCLUDE}/ovlpjoin.cc
                           ${COMMON_INCLUDE}/ovlpjoin_parallel.cc
                           ${COMMON_INCLUDE}/simfunc.cc)
    target_include_directories(overlap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../cpp)
elseif(EXECUTABLE_NAME STREQUAL "str")
    add_executable(str test_stringjoin.cc 
                       ${COMMON_INCLUDE}/io.cc
                       ${COMMON_INCLUDE}/dataframe.cc
                       ${COMMON_INCLUDE}/tokenizer.cc
                       ${COMMON_INCLUDE}/stringjoin.cc 
                       ${COMMON_INCLUDE}/stringjoin_parallel.cc
                       ${COMMON_INCLUDE}/joinutil.cc
                       ${COMMON_INCLUDE}/simfunc.cc)
    target_include_directories(str PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../cpp)
else()
    message(FATAL_ERROR "Invalid executable name specified: ${EXECUTABLE_NAME}")
endif()