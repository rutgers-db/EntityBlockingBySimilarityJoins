set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../../../shared_lib)
add_library(block SHARED ../common/io.cc 
                         ../common/tokenizer.cc 
                         ../common/dataframe.cc 
                         ../common/setjoin.cc
                         ../common/setjoin_parallel.cc 
                         ../common/ovlpjoin.cc
                         ../common/ovlpjoin_parallel.cc 
                         ../common/stringjoin.cc 
                         ../common/stringjoin_parallel.cc 
                         ../common/joinutil.cc 
                         ../common/simfunc.cc 
                         ../topk/topk.cc  
                         block.cc
                         blocker_config.cc
                         blocker_util.cc
                         simjoin_blocker.cc 
                         knn_blocker.cc)
target_include_directories(block PRIVATE ..)
# target_link_libraries(block PRIVATE arrow_static parquet_static)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../../bin)
add_executable(block_app block_main.cc)
target_include_directories(block_app PRIVATE ..)
target_link_libraries(block_app PRIVATE block)

set_property(TARGET block_app PROPERTY OUTPUT_NAME block)