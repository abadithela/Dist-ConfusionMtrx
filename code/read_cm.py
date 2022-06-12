# File to read confusion matrix:
# Apurva Badithela
# June 11, 2022
import os
import pdb
from utils import *
dataroot="/Users/apurvabadithela/Documents/software/nuscenes/data/sets/nuscenes"
dataroot_ext = "/Volumes/Extreme SSD/nuscenes"
import pickle as pkl
from parse_matchings import ConfusionMatrix, cluster_categories
import numpy as np
from itertools import chain, combinations
import pdb

save_data_ext = False
cm_type = "prop_based"
horizon = 100
discrete_horizon = 10
distance_bins = int(horizon/discrete_horizon)
# Get number of classes in the confusion matrix:
# Nclasses: num of different objects
# Nprops: num of different confusion matrices

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def get_nclasses(Nclasses):
    if cm_type == "prop_based":
        return 2**(Nclasses-1)
    else:
        return Nclasses

def get_prop_cm_classes(class_dict):
    n = len(list(class_dict.keys()))
    Nclasses = get_nclasses(n)
    cm_class_dict = dict()
    cm_class_dict[Nclasses-1] = {'empty'} # Building an order
    key = 0
    empty_key = [k for k,v in class_dict.items() if v == 'empty'][0] # There should only be one empty key
    class_keys = list(class_dict.keys())[:-1] # Fix; make better
    cm_classes_keys = list(powerset(class_keys))

    for k in range(len(cm_classes_keys)):
        class_combinations = cm_classes_keys[k]
        cm_class_dict[k] = {class_dict[class_combinations[0]]}
        for l in range(1, len(class_combinations)):
            cm_class_dict[k] |= {class_dict[class_combinations[l]]}
    return cm_class_dict


def read_from_file(fname=None):
    if fname is None:
        if save_data_ext:
            dirname = "/Volumes/Extreme SSD/cm_processing/" + traindir + "/"
        else:
            dirname = cwd + "/"
        fname = dirname + "cm/prop_label_hrzn_cluster_cat" + str(horizon) + "_distbin_" + str(distance_bins) + ".p"

    conf_matrix = pkl.load(open(fname, "rb"))
    return conf_matrix

if __name__=="__main__":
    class_dict = {0: 'pedestrian', 1:'obstacle', 2:'empty'}
    cm_class_dict = get_prop_cm_classes(class_dict)
    classes = [tuple(c) for c in list(cm_class_dict.values()) if c!={'empty'}]
    C = ConfusionMatrix(classes, horizon, distance_bins=distance_bins)
    cur_dir = os.getcwd()
    fname = cur_dir + "/cm/prop_label_hrzn_cluster_cat100_distbin_10.p"
    conf_matrix = read_from_file(fname=fname)
    C.C = conf_matrix
    C.print_cm()
    pdb.set_trace()
