#!/usr/bin/python
# -*- coding: utf-8 -*-

# file with utility variables & functions

import datetime as dt
import numpy as np

from itertools import permutations, product
from scipy.stats import pearsonr

logs_start_day = dt.datetime.strptime("2015-05-17", '%Y-%m-%d')

# this should be 10 for the actual experiments (or 1 for the test with the sample file)
num_tests = 10

# set here window length, training window length and testing window length
window_length = 6
train_w_length = 5
test_w_length = 1

# directory where the data are stored 
data_dir = '../data/' 
data_prefix = 'df_sample_'

# get the gub prediction - i.e. blacklist is the union of blacklists for all contributors in the cluster
def gub_prediction(contributors, blacklists, whitelists):
    
    gub_b = set()
    gub_w = set()
        
    for contributor in contributors:
        gub_b = gub_b | blacklists[contributor]
            
    for contributor in contributors:
        gub_w = gub_w | whitelists[contributor]
    
    # the global whitelist is the union of local whitelists - the union of local blacklists    
    gub_w = gub_w - gub_b        
        
    return gub_b, gub_w        

# for each contributor, get the intersection of the attackers in his training set with the blacklist 
# of the rest contributors in the cluster
# then, get the union with the local blacklist     
def intersection_prediction(contributor, contributors, blacklists, whitelists, train_set_attackers):    
    
    int_bl = set()
    int_wl = set()
    
    # if the cluster has one contributor then the local blacklist is returned
    if len(contributors) == 1:
        int_bl = blacklists[contributor]
    
    else:
        for cont in contributors:
            int_bl = int_bl | (train_set_attackers[contributor] & blacklists[cont])
        
        int_bl = blacklists[contributor] | int_bl
    
    # the whitelist is the local whitelist - the intersection blacklist
    int_wl = whitelists[contributor] - int_bl
        
    return int_bl, int_wl

# compute some prediction stats
def verify_prediction(local_blacklist, local_whitelist, gub_blacklist, gub_whitelist, int_blacklist, int_whitelist, ground_truth):

    assert type(local_blacklist) is set
    assert type(local_whitelist) is set
    assert type(gub_blacklist) is set
    assert type(gub_whitelist) is set
    assert type(int_blacklist) is set
    assert type(int_whitelist) is set
    assert type(ground_truth) is set

    d = {}
    d["tp_local"] = len( local_blacklist & ground_truth )
    d["fp_local"] = len( local_blacklist - ground_truth )
    d["fn_local"] = len( local_whitelist & ground_truth )
    d["tn_local"] = len( local_whitelist - ground_truth )
    
    d["tp_gub"] = len( gub_blacklist & ground_truth )
    d["fp_gub"] = len( gub_blacklist - ground_truth )
    d["fn_gub"] = len( gub_whitelist & ground_truth )
    d["tn_gub"] = len( gub_whitelist - ground_truth )
    
    d["tp_int"] = len( int_blacklist & ground_truth )
    d["fp_int"] = len( int_blacklist - ground_truth )
    d["fn_int"] = len( int_whitelist & ground_truth )
    d["tn_int"] = len( int_whitelist - ground_truth )        
        
    d["n_attacks"] = len(ground_truth)         
        
    try:
        d["tp_impr_gub"] = (d["tp_gub"] - d["tp_local"]) / float(d["tp_local"])
        d["tp_impr_int"] = (d["tp_int"] - d["tp_local"]) / float(d["tp_local"])
    except ZeroDivisionError:
        d["tp_impr_gub"] = 0.0
        d["tp_impr_int"] = 0.0
        
    try:    
        d["fp_incr_gub"] = (d["fp_gub"] - d["fp_local"]) / float(d["fp_local"])
        d["fp_incr_int"] = (d["fp_int"] - d["fp_local"]) / float(d["fp_local"])    
    except ZeroDivisionError:
        d["fp_incr_gub"] = 0.0
        d["fp_incr_int"] = 0.0
        
    return d