cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -fopenmp -mavx2 -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -fopenmp -mavx2")
SET(CMAKE_ALL_ERRORS_FATAL TRUE)

project(simjoin_entitymatching)

# find_package(Arrow REQUIRED)
# find_package(Parquet REQUIRED)

add_subdirectory(cpp/blocker)
add_subdirectory(cpp/sampler)
add_subdirectory(cpp/feature)