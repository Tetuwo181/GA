#coding: UTF-8
import numpy as np
from ..indi import individual as INDI
from numba import jit 

#上限と下限をカットする
def cut_upper_and_under(genom, max_base = None, min_base = None):
    if max_base is None:
        max_list = np.array([1.0 for index in range(len(genom[:]))])
    else:
        max_list = max_base
    if min_base is None:
        min_list = np.array([0 for index in range(len(genom[:]))])
    else:
        min_list = min_base
    under_cutted = np.maximum(genom, min_list)
    upper_under_cutted = np.minimum(under_cutted, max_list)
    return upper_under_cutted

def set_max_and_min(max_list = None, min_list = None):
    if max_list is None:
        max_vec = np.array([1.0 for index in range(len(max_list))])
    else:
        max_vec = np.array(max_list)
    if min_list is None:
        min_vec = np.array([0 for index in range(len(max_list))])
    else:
        min_vec = np.array(min_list)
    range_max_and_min = np.abs(max_vec-min_vec)
    return (max_vec, min_vec, range_max_and_min)    
    

def build_max_min_list(length, max_num, min_num):
    min_base = [min_num for index in range(length)]
    max_base = [max_num for index in range(length)]
    return (np.array(max_base), np.array(min_base))


