# experiments utils
from os import system
from typing import Literal


def cat_blocking_topk_output_first(dataname, dtype, turn=Literal[1, 2, 3]):
    topk_intermedia = "output/topk_stat/intermedia.txt"
    topk_exp_log = "output/topk_stat/" + dataname + "_" + dtype + ".txt"
    print(f"--- report top k blocking result on turn {turn} ---", flush=True)
    echo_command = "echo -e \"\n this is an experiment\" >> " + topk_exp_log
    system(echo_command)
    cat_command = "cat " + topk_intermedia + " >> " + topk_exp_log
    system(cat_command)


def cat_blocking_topk_output_second(dataname, dtype, rep_attr, turn=Literal[1, 2, 3]):
    topk_intermedia = "output/topk_stat/intermedia.txt"
    topk_exp_log = "output/topk_stat/" + dataname + "_" + dtype + ".txt"
    cat_command = "cat " + topk_intermedia + " >> " + topk_exp_log
    system(cat_command)
    echo_command = "echo -" + str(turn) + " \"\n\" >> " + topk_exp_log
    system(echo_command)
    echo_command = "echo " + rep_attr + " >> " + topk_exp_log
    system(echo_command)


def cat_match_res_output_first(dataname, dtype, turn=Literal[1, 2, 3]):
    match_intermedia = "output/match_stat/intermedia.txt"
    match_exp_log = "output/match_stat/" + dataname + "_" + dtype + ".txt"
    print(f"--- report matching result on turn {turn} ---", flush=True)
    echo_command = "echo -e \"\n this is an experiment\" >> " + match_exp_log
    system(echo_command)
    cat_command = "cat " + match_intermedia + " >> " + match_exp_log
    system(cat_command)


def cat_match_res_output_second(dataname, dtype, turn=Literal[1, 2, 3]):
    match_intermedia = "output/match_stat/intermedia.txt"
    match_exp_log = "output/match_stat/" + dataname + "_" + dtype + ".txt"
    cat_command = "cat " + match_intermedia + " >> " + match_exp_log
    system(cat_command)
    echo_command = "echo -" + str(turn) + " \"\n\" >> " + match_exp_log
    system(echo_command)
