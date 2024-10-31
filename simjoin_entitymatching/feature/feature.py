# author: Yunqi Li
# contact: liyunqixa@gmail.com
import pathlib
import subprocess
from ctypes import c_bool, c_int, c_char, POINTER, c_char_p, byref, Structure
from tkinter import StringVar, Tk
from ctypes import cdll
import numpy as np
from typing import Literal
from simjoin_entitymatching.feature.feature_base import NewFeatureExtractor
import py_entitymatching as em


def run_command(cmd_args, log_file):
    try:
        cmd_res = subprocess.run(cmd_args, stdout=log_file, stderr=log_file, 
                                 check=True)
    except subprocess.CalledProcessError as cmd_err:
        print(f"error! exit with {cmd_err.returncode}")
        print(f"message {cmd_err.output}")
        raise
    else:
        print(f"feature command exit with {cmd_res.returncode}")


def feature_binary_decorator(command):
    def wrapper(*args, **kwargs):
        print("running feature binary files")
        command(*args, **kwargs)
        print("done!")
    return wrapper


@feature_binary_decorator
def run_feature_bin(is_interchangeable, flag_consistent, total_table, total_attr, attrs, usage=Literal["match", "topk"]):
    cur_file_dir = str(pathlib.Path(__file__).parent.resolve())
    feature_bin_path = "/".join([cur_file_dir, "bin", "feature"])
    feature_args = [usage, str(is_interchangeable), str(flag_consistent), str(total_table), str(total_attr)]
    feature_args.extend(attrs)
    cmd_args = [feature_bin_path]
    cmd_args.extend(feature_args)
    
    feature_output_file = "/".join([cur_file_dir, "build", "feature_output.log"])
    with open(feature_output_file, "w") as feature_log:
        run_command(cmd_args, feature_log)
        
        
def feature_library_decorator(command):
    def wrapper(*args, **kwargs):
        print("running feature compiling commands")
        command(*args, **kwargs)
        print("done!")
    return wrapper


@feature_library_decorator
def compile_feature(mode):
    cur_file_dir = str(pathlib.Path(__file__).parent.resolve())
    cur_mode = "".join(["-Dcompile_mode=", mode])
    src_dir = "/".join([cur_file_dir, "cpp"])
    build_dir = "/".join([cur_file_dir, "build"])
    nohup_compile_output = "/".join([build_dir, "compile.log"])
    
    with open(nohup_compile_output, "w") as compile_log:
        print("set up compiling requirements...")
        specify_build_dir = ["cmake", cur_mode, "-DCMAKE_BUILD_TYPE=Release", "-S", src_dir, "-B", build_dir]
        run_command(specify_build_dir, compile_log)
        
        print("compiling feature...")
        build_feature = ["cmake", "--build", build_dir, "--clean-first"]
        run_command(build_feature, compile_log)
    
    
@feature_library_decorator
def run_feature_lib(is_interchangeable, flag_consistent, total_table, total_attr, attrs, usage=Literal["match", "topk"], 
                    default_fea_vec_dir="", default_icv_dir="", default_fea_names_dir=""):
    # load
    cur_file_dir = str(pathlib.Path(__file__).parent.resolve())
    feature_lib_path = "/".join([cur_file_dir, "..", "..", "shared_lib", "libfeature.so"])
    feature_lib = cdll.LoadLibrary(feature_lib_path)
    
    class FeatureArguments(Structure):
        _fields_ = [
            ("total_attr", c_int), 
            ("attributes", c_char_p * 20)
        ]
    
    fa = FeatureArguments()
    fa.total_attr = total_attr
    for i in range(total_attr):
        cur_attr = attrs[i]
        fa.attributes[i] = cur_attr.encode('utf-8')
    default_fea_vec_dir = default_fea_vec_dir.encode('utf-8')
    default_icv_dir = default_icv_dir.encode('utf-8')
    default_fea_names_dir = default_fea_names_dir.encode('utf-8')
    
    '''
    C api:
        void extract_features_4_matching(int is_interchangeable, bool flag_consistent, int total_table, 
                                         const FeatureArguments *attrs, const char *default_fea_vec_dir)
        void extract_features_4_topk(int is_interchangeable, bool flag_consistent, int total_table, 
                                     const FeatureArguments *attrs, const char *default_fea_vec_dir) 
    '''
    
    if usage == "match":
        feature_lib.extract_features_4_matching.argtypes = [c_int, c_bool, c_int, POINTER(FeatureArguments), c_char_p, c_char_p, c_char_p]
        feature_lib.extract_features_4_matching.restype = None
        
        feature_lib.extract_features_4_matching(is_interchangeable, flag_consistent, total_table, byref(fa), default_fea_vec_dir, 
                                                default_icv_dir, default_fea_names_dir)
    elif usage == "topk":
        feature_lib.extract_features_4_topk.argtypes = [c_int, c_bool, c_int, POINTER(FeatureArguments), c_char_p, c_char_p, c_char_p]
        feature_lib.extract_features_4_topk.restype = None
        
        feature_lib.extract_features_4_topk(is_interchangeable, flag_consistent, total_table, byref(fa), default_fea_vec_dir, 
                                            default_icv_dir, default_fea_names_dir)
        
        
def run_feature_megallen(tableA, tableB, feature_tab, total_table, is_interchangeable=Literal[0, 1], flag_consistent=Literal[0, 1], 
                         attrs_after=None, group=None, cluster=None, default_blk_res_dir="", n_jobs=1):
    '''
    a wrapper of the extract_fea_vec
    '''
    cur_file_dir = str(pathlib.Path(__file__).parent.resolve())
    if default_blk_res_dir == "":
        blk_res_dir = "/".join([cur_file_dir, "..", "..", "output", "blk_res"])
    else:
        default_blk_res_dir = default_blk_res_dir[ : -1] if default_blk_res_dir[-1] == '/' \
                                                         else default_blk_res_dir
        blk_res_dir = default_blk_res_dir
    
    for i in range(total_table):
        blk_res_path = "/".join([blk_res_dir, "blk_res" + str(i) + ".csv"])
        blk_res_cand = em.read_csv_metadata(blk_res_path, key="_id", 
                                            ltable=tableA, rtable=tableB, 
                                            fk_ltable="ltable_id", 
                                            fk_rtable="rtable_id")
        
        if is_interchangeable == 0:
            H = em.extract_feature_vecs(blk_res_cand, 
                                        feature_table=feature_tab, 
                                        attrs_after=attrs_after,
                                        show_progress=False, 
                                        n_jobs=n_jobs)
            fea_vec_path = "/".join([blk_res_dir, "feature_vec" + str(i) + "_py.csv"])
        else:
            H = NewFeatureExtractor.extract_feature_vecs(blk_res_cand, feature_table=feature_tab, 
                                                         attrs_after=attrs_after, 
                                                         group=group, cluster=cluster, 
                                                         n_jobs=n_jobs)
            fea_vec_path = "/".join([blk_res_dir, "feature_vec" + str(i) + ".csv"])
        
        H.to_csv(fea_vec_path, index=False)