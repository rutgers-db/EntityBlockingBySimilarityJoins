set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../../../shared_lib)
add_library(sample SHARED ../common/io.cc 
                          ../common/tokenizer.cc 
                          ../common/dataframe.cc 
                          ../common/setjoin_parallel.cc 
                          ../common/ovlpjoin_parallel.cc 
                          ../common/joinutil.cc 
                          ../common/simfunc.cc 
                          sample.cc 
                          sampler_impl.cc 
                          sampler_util.cc)
target_include_directories(sample PRIVATE ..)
# target_link_libraries(sample PRIVATE arrow_static parquet_static)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../../bin)
add_executable(sample_app sample_main.cc)
target_include_directories(sample_app PRIVATE ..)
target_link_libraries(sample_app PRIVATE sample)

set_property(TARGET sample_app PROPERTY OUTPUT_NAME sample)