# author: Yunqi Li
# contact: liyunqixa@gmail.com
import pathlib
import subprocess
from ctypes import c_char_p, c_double, c_bool
from ctypes import cdll
from typing import Literal


def run_command(cmd_args, log_file):
    try:
        cmd_res = subprocess.run(cmd_args, stdout=log_file, stderr=log_file, 
                                 check=True)
    except subprocess.CalledProcessError as cmd_err:
        print(f"error! exit with {cmd_err.returncode}")
        print(f"message {cmd_err.output}")
        raise
    else:
        print(f"group command exit with {cmd_res.returncode}")


def group_binary_decorator(command):
    def wrapper(*args, **kwargs):
        print("running group binary files")
        command(*args, **kwargs)
        print("done!")
    return wrapper


@group_binary_decorator
def run_group_bin(group_attribute, group_strategy, group_tau, is_transitive_closure=Literal[0, 1], default_icv_dir=""):
    group_args = [group_attribute, group_strategy, str(group_tau), str(is_transitive_closure), default_icv_dir]
    
    cur_file_dir = str(pathlib.Path(__file__).parent.resolve())
    group_bin_path = "/".join([cur_file_dir, "bin", "group"])
    cmd_args = [group_bin_path]
    cmd_args.extend(group_args)
    
    group_output_file = "/".join([cur_file_dir, "build", "group_output.log"])
    with open(group_output_file, "w") as group_log:
        run_command(cmd_args, group_log)
        
        
def group_library_decorator(command):
    def wrapper(*args, **kwargs):
        print("running group library files")
        command(*args, **kwargs)
        print("done!")
    return wrapper


@group_library_decorator
def compile_group(mode):
    cur_file_dir = str(pathlib.Path(__file__).parent.resolve())
    cur_mode = "".join(["-Dcompile_mode=", mode])
    src_dir = "/".join([cur_file_dir, "cpp"])
    build_dir = "/".join([cur_file_dir, "build"])
    nohup_compile_output = "/".join([build_dir, "compile.log"])
    
    with open(nohup_compile_output, "w") as compile_log:
        print("set up compiling requirements...")
        specify_build_dir = ["cmake", cur_mode, "-DCMAKE_BUILD_TYPE=Release", "-S", src_dir, "-B", build_dir]
        run_command(specify_build_dir, compile_log)
        
        print("compiling group...")
        build_group = ["cmake", "--build", build_dir, "--clean-first"]
        run_command(build_group, compile_log)
    
    
@group_library_decorator
def run_group_lib(group_attribute, group_strategy, group_tau, is_transitive_closure=Literal[0, 1], default_icv_dir=""):
    # load
    cur_file_dir = str(pathlib.Path(__file__).parent.resolve())
    group_lib_path = "/".join([cur_file_dir, "..", "..", "shared_lib", "libgroup.so"])
    group_lib = cdll.LoadLibrary(group_lib_path)
    
    group_attribute = group_attribute.encode('utf-8')
    group_strategy = group_strategy.encode('utf-8')
    default_icv_dir = default_icv_dir.encode('utf-8')
    
    if group_strategy == "graph":
        group_lib.group_interchangeable_values_by_graph.argtypes = [c_char_p, c_char_p, c_double, c_bool, c_char_p]
        group_lib.group_interchangeable_values_by_graph.restype = None
        group_lib.group_interchangeable_values_by_graph(group_attribute, group_strategy, group_tau, c_bool(is_transitive_closure), 
                                                        default_icv_dir)
    elif group_strategy == "cluster":
        pass
    else:
        print(f"no such group strategy : {group_strategy}")