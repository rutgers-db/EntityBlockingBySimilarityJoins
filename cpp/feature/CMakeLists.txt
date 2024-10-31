set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../../../shared_lib)
add_library(feature SHARED ../common/io.cc 
                           ../common/tokenizer.cc 
                           ../common/dataframe.cc 
                           cal_feature.cc
                           feature_index.cc
                           feature_utils.cc
                           feature.cc)
target_include_directories(feature PRIVATE ..)
# target_link_libraries(feature PRIVATE arrow_static parquet_static)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../../bin)
add_executable(feature_app feature_main.cc)
target_include_directories(feature_app PRIVATE ..)
target_link_libraries(feature_app PRIVATE feature)

set_property(TARGET feature_app PROPERTY OUTPUT_NAME feature)