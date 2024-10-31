# author: Yunqi Li
# contact: liyunqixa@gmail.com
import pathlib
import subprocess
from ctypes import c_char_p, c_double, c_uint, c_bool, c_int
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
        print(f"sample command exit with {cmd_res.returncode}")


def sample_binary_decorator(command):
    def wrapper(*args, **kwargs):
        print("running sample binary files")
        command(*args, **kwargs)
        print("done!")
    return wrapper


@sample_binary_decorator
def run_sample_bin(sample_strategy, blocking_attr, cluster_tau, sample_tau, step2_tau, num_data):
    sample_settings = {
						"title": {"cluster": [str(cluster_tau), str(sample_tau), str(step2_tau), "", "", ""], "down": ["100000", "20", "", "", ""], "pre": ["1000000", "1", "", ""]}, 
						"name": {"cluster": [str(cluster_tau), str(sample_tau), str(step2_tau), "", "", ""], "down": ["100000", "20", "", "", ""], "pre": ["1000000", "1", "", ""]}
					  }
    args_dict = sample_settings[blocking_attr]
    
    sample_args = [blocking_attr, str(num_data), sample_strategy]
    sample_args.extend(args_dict[sample_strategy])
    
    cur_file_dir = str(pathlib.Path(__file__).parent.resolve())
    sample_bin_path = "/".join([cur_file_dir, "bin", "sample"])
    cmd_args = [sample_bin_path]
    cmd_args.extend(sample_args)
    
    sample_output_file = "/".join([cur_file_dir, "build", "sample_output.log"])
    with open(sample_output_file, "w") as sample_log:
        run_command(cmd_args, sample_log)
        
        
def sample_library_decorator(command):
    def wrapper(*args, **kwargs):
        print("running sample compiling commands")
        command(*args, **kwargs)
        print("done!")
    return wrapper


@sample_library_decorator
def compile_sample(mode):
    cur_file_dir = str(pathlib.Path(__file__).parent.resolve())
    cur_mode = "".join(["-Dcompile_mode=", mode])
    src_dir = "/".join([cur_file_dir, "cpp"])
    build_dir = "/".join([cur_file_dir, "build"])
    nohup_compile_output = "/".join([build_dir, "compile.log"])
    
    with open(nohup_compile_output, "w") as compile_log:
        print("set up compiling requirements...")
        specify_build_dir = ["cmake", cur_mode, "-DCMAKE_BUILD_TYPE=Release", "-S", src_dir, "-B", build_dir]
        run_command(specify_build_dir, compile_log)
        
        print("compiling sample...")
        build_sample = ["cmake", "--build", build_dir, "--clean-first"]
        run_command(build_sample, compile_log)
    
    
@sample_library_decorator
def run_sample_lib(sample_strategy: Literal["cluster", "down", "pre"], blocking_attr, cluster_tau, sample_tau, step2_tau, num_data, 
                   path_table_A="", path_table_B="", path_default_output_dir=""):
    # load
    cur_file_dir = str(pathlib.Path(__file__).parent.resolve())
    sample_lib_path = "/".join([cur_file_dir, "..", "..", "shared_lib", "libsample.so"])
    sample_lib = cdll.LoadLibrary(sample_lib_path)
    
    blocking_attr = blocking_attr.encode('utf-8')
    default_path_table_A = path_table_A.encode('utf-8')
    default_path_table_B = path_table_B.encode('utf-8')
    path_default_output_dir = path_default_output_dir.encode('utf-8')
    
    if sample_strategy == "cluster":
        if num_data == 1:
            sample_lib.cluster_sample_self.argtypes = [c_char_p, c_double, c_double, c_char_p, c_char_p, c_char_p]
            sample_lib.cluster_sample_self.restype = None
            sample_lib.cluster_sample_self(blocking_attr, cluster_tau, sample_tau, default_path_table_A, default_path_table_B, 
                                           path_default_output_dir)
        elif num_data == 2:
            sample_lib.cluster_sample_RS.argtypes = [c_char_p, c_double, c_double, c_double, c_char_p, c_char_p, c_char_p]
            sample_lib.cluster_sample_RS.restype = None
            sample_lib.cluster_sample_RS(blocking_attr, cluster_tau, sample_tau, step2_tau, default_path_table_A, default_path_table_B, 
                                         path_default_output_dir)
    elif sample_strategy == "down":
        sample_lib.down_sample.argtypes = [c_uint, c_uint, c_char_p, c_bool, c_char_p, c_char_p, c_char_p]
        sample_lib.down_sample.restype = None
        is_RS = c_bool(num_data == 2)
        sample_lib.down_sample(c_uint(100000), c_uint(20), blocking_attr, is_RS, default_path_table_A, default_path_table_B, 
                               path_default_output_dir)
    elif sample_strategy == "pre":
        sample_lib.pre_sample.argtypes = [c_uint, c_int, c_char_p, c_char_p, c_char_p]
        sample_lib.pre_sample.restype = None
        sample_lib.pre_sample(c_uint(1000000), c_int(1), blocking_attr, default_path_table_A, default_path_table_B)