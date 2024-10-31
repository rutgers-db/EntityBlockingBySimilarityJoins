# author: Yunqi Li
# contact: liyunqixa@gmail.com
import pathlib
import subprocess
from ctypes import c_char_p, c_double, c_uint, c_bool, c_int, c_uint64
from ctypes import cdll
import numpy as np
from typing import Literal
import simjoin_entitymatching.blocker.extract_formula as exf


def extract_block_rules(trigraph, rule_path, move_strategy=Literal["basic", "greedy"], 
                        additional_rule_path=None, optimal_rule_path=None):
    ef = exf.ExtractFormula()
    ms = 0 if move_strategy == "basic" else 1
    ef.extract(trigraph, move_strategy=ms)
    print("done", flush=True)
    ef.flush_rules(trigraph, rule_path)
    if additional_rule_path is not None:
        ef.get_rules_cur_comb(trigraph, additional_rule_path)
    if optimal_rule_path is not None:
        ef.get_optimal_rules_comb(trigraph, optimal_rule_path)

    # select
    # selected = ef.select_partial(trigraph, short_numeric_attr)
    # ef.flush_rules(trigraph, selected)


def run_command(cmd_args, log_file):
    try:
        cmd_res = subprocess.run(cmd_args, stdout=log_file, stderr=log_file, 
                                 check=True)
    except subprocess.CalledProcessError as cmd_err:
        print(f"error! exit with {cmd_err.returncode}")
        print(f"message {cmd_err.output}")
        raise
    else:
        print(f"block command exit with {cmd_res.returncode}")


def block_binary_decorator(command):
    def wrapper(*args, **kwargs):
        print("running block binary files")
        command(*args, **kwargs)
        print("done!")
    return wrapper


@block_binary_decorator
def run_block_bin(blocking_attr, blocking_attr_type, blocking_top_k, path_tableA, path_tableB, path_gold, path_rule, 
                  table_size, is_join_topk, is_idf_weighted, num_data, path_default_output_dir="", 
                  path_default_sample_res=""):
    jt = num_data - 1
    js = 1 if num_data == 1 else 0
    
    cur_file_dir = str(pathlib.Path(__file__).parent.resolve())
    block_bin_path = "/".join([cur_file_dir, "bin", "block"])
    block_args = [str(jt), str(js), str(blocking_top_k), blocking_attr, blocking_attr_type, 
                  path_tableA, path_tableB, path_gold, path_rule, str(table_size), str(is_join_topk), str(is_idf_weighted), 
                  path_default_output_dir, path_default_sample_res]
    cmd_args = [block_bin_path]
    cmd_args.extend(block_args)
    
    block_output_file = "/".join([cur_file_dir, "build", "block_output.log"])
    with open(block_output_file, "w") as block_log:
        run_command(cmd_args, block_log)
        
        
def block_library_decorator(command):
    def wrapper(*args, **kwargs):
        print("running block compiling commands")
        command(*args, **kwargs)
        print("done!")
    return wrapper


@block_library_decorator
def compile_block(mode):
    cur_file_dir = str(pathlib.Path(__file__).parent.resolve())
    cur_mode = "".join(["-Dcompile_mode=", mode])
    src_dir = "/".join([cur_file_dir, "cpp"])
    build_dir = "/".join([cur_file_dir, "build"])
    nohup_compile_output = "/".join([build_dir, "compile.log"])
    
    with open(nohup_compile_output, "w") as compile_log:
        print("set up compiling requirements...")
        specify_build_dir = ["cmake", cur_mode, "-DCMAKE_BUILD_TYPE=Release", "-S", src_dir, "-B", build_dir]
        run_command(specify_build_dir, compile_log)
        
        print("compiling block...")
        build_block = ["cmake", "--build", build_dir, "--clean-first"]
        run_command(build_block, compile_log)
    
    
@block_library_decorator
def run_simjoin_block_lib(blocking_attr, blocking_attr_type, blocking_top_k, path_tableA, path_tableB, path_gold, path_rule, 
                          table_size, is_join_topk = Literal[0, 1], is_idf_weighted = Literal[0, 1], num_data = Literal[1, 2], 
                          path_default_output_dir="", path_default_sample_res=""):
    # load
    cur_file_dir = str(pathlib.Path(__file__).parent.resolve())
    block_lib_path = "/".join([cur_file_dir, "..", "..", "shared_lib", "libblock.so"])
    block_lib = cdll.LoadLibrary(block_lib_path)
    
    jt = num_data - 1
    js = 1 if num_data == 1 else 0
    blocking_top_k = np.uint64(int(blocking_top_k))
    blocking_attr = blocking_attr.encode('utf-8')
    blocking_attr_type = blocking_attr_type.encode('utf-8')
    path_table_A = path_tableA.encode('utf-8')
    path_table_B = path_tableB.encode('utf-8')
    path_gold = path_gold.encode('utf-8')
    path_rule = path_rule.encode('utf-8')
    table_size = int(table_size)
    is_join_topk = bool(is_join_topk)
    is_idf_weighted = bool(is_idf_weighted)
    path_default_output_dir = path_default_output_dir.encode('utf-8')
    path_default_sample_res = path_default_sample_res.encode('utf-8')
    
    # C api:
    # void sim_join_block(int join_type, int join_setting, uint64_t top_k, const char *topk_attr, const char *attr_type, 
	# 					  const char *path_table_A, const char *path_table_B, const char *path_gold, 
	# 					  const char *path_rule, int table_size, bool is_join_topk, bool is_idf_weighted)
 
    block_lib.sim_join_block.argtypes = [c_int, c_int, c_uint64, c_char_p, c_char_p, c_char_p, c_char_p, c_char_p, 
                                         c_char_p, c_int, c_bool, c_bool, c_char_p, c_char_p]
    block_lib.sim_join_block.restype = None
    
    block_lib.sim_join_block(jt, js, blocking_top_k, blocking_attr, blocking_attr_type, 
                             path_table_A, path_table_B, path_gold, 
                             path_rule, table_size, is_join_topk, is_idf_weighted, 
                             path_default_output_dir, path_default_sample_res)
    
    
@block_library_decorator
def run_knn_block_lib(blocking_attr, blocking_attr_type, blocking_top_k, path_tableA, path_tableB, path_gold, path_rule, 
                      table_size, is_join_topk, is_idf_weighted, num_data):
    pass 