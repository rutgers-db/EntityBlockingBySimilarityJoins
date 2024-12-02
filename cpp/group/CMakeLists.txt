set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../../../shared_lib)
add_library(group SHARED group.cc
                         graph.cc
                         cluster.cc)
target_include_directories(group PRIVATE ..)
# target_link_libraries(group PRIVATE arrow_static parquet_static)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../../bin)
add_executable(group_app group_main.cc)
target_include_directories(group_app PRIVATE ..)
target_link_libraries(group_app PRIVATE group)

set_property(TARGET group_app PROPERTY OUTPUT_NAME group)