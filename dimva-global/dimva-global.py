#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '../utils/')
from dimva_util import *

import time_series as ts

import numpy as np
import pandas as pd

from itertools import combinations
from itertools import product

from scipy.stats import itemfreq
from scipy.sparse import lil_matrix

from collections import Counter

stats_list = []
nn_ips = 50
total_pairs = (70 * 69) / 2
percentage = [1, 2, 3, 4, 5]

kNN_alg = ['auto', 'ball_tree', 'kd_tree', 'brute']
    
for i in range(0, num_tests):
    
    print 'Window: ', i
    start_day = logs_start_day + dt.timedelta(days=i)
    
    # load the window data into a dataframe
    
    # sample file for testing purposes
    # window_logs = pd.read_pickle(data_dir + "sample.pkl")
    
    # data for the experiments
    window_logs = pd.read_pickle(data_dir + data_prefix + start_day.date().isoformat() + ".pkl")

    # extract /24 subnets from IPs
    window_logs.src_ip = window_logs.src_ip.map(lambda x: x[:11])
    
    # get the contributors of the window logs
    top_targets = np.unique( window_logs["target_ip"] )

    # get the days, as well as first day and last day
    days = np.unique(window_logs['D'])
    first_day, last_day = np.min(days), np.max(days)

    # split training set and testing set
    train_date_list = [start_day.date() + dt.timedelta(days=x) for x in range(0, window_length - test_w_length)]
    train_set = window_logs[window_logs.D.isin(train_date_list)] 
    
    test_date_list = [start_day.date() + dt.timedelta(days=x) for x in range(train_w_length, window_length)]
    test_set = window_logs[window_logs.D.isin(test_date_list)]
    
    print 'Train dates: ', train_set['D'].min(), train_set['D'].max()
    print 'Test dates: ', test_set['D'].min(), test_set['D'].max()
    print 'Training set size: ', train_set.shape[0]
    print 'Test set size: ', test_set.shape[0]

    del window_logs
    
    # get the pairs between organizations 
    target_pairs = combinations(top_targets, 2)

    # dictionary holding the indices of the contributors
    ind_dic = dict( zip(top_targets, range(top_targets.size) ))

    # organization to organization matrix
    o2o = np.zeros((top_targets.size, top_targets.size, len(train_date_list) ))

    # create a dictionary where each contributors stores his attacker set over all the training window
    print 'creating attacker set dictionary...'
    victim_set = dict()
    for target in top_targets:
        victim_set[target] = set( train_set[ (train_set.target_ip == target) ].src_ip )     
    
    # create a dictionary where each contributor stores his attacker set for each day 
    print 'Creating daily attacker set dictionary...'
    victim_daily_set = dict()
    for target in top_targets:
        victim_daily_set[target] = dict()
        for idx, day in enumerate(train_date_list):
            victim_daily_set[target][idx] = set( train_set[ (train_set.target_ip == target) & (train_set.D == day) ].src_ip)     

    # create the organisation to organisation matrix with 
    # TODO: compute some other form of similarity e.g. Jaccard, Cosine
    print 'Creating o2o matrix...'
    for pair in target_pairs:
        for idx, day in enumerate(train_date_list):
            o2o[ ind_dic[pair[0]], ind_dic[pair[1]], idx] = len( victim_daily_set[pair[0]][idx] & victim_daily_set[pair[1]][idx])
            o2o[ ind_dic[pair[1]], ind_dic[pair[0]], idx] = o2o[ind_dic[pair[0]], ind_dic[pair[1]], idx]

    # local prediction and blacklist generation part - this dictionary holds each contributor's local blacklist       
    print 'Computing local predictions...'
    l_blacklists = dict()
    l_whitelists = dict()
    l_blacklists, l_whitelists = ts.local_prediction(top_targets, train_set, i)      
    
    # dimva clustering part 
    for perc in percentage:
        
        print 'Percentage : ', perc
        pairs = (total_pairs * perc) / 100
        print 'Number of pairs : ', pairs
        
        X = o2o.sum(axis=2)
        X_1d = X.flatten()
        idx_1d = X_1d.argsort()[-pairs*2:]
    
        # find the indexes of the top pairs
        x_idx, y_idx = np.unravel_index(idx_1d, X.shape)
        clusters = [(top_targets[x_idx[i]], top_targets[y_idx[i]]) for i in range(0, pairs*2)]
        #clusters = [(x_idx[i], y_idx[i]) for i in range(0,pairs)]
        clusters = set(clusters)
        clusters = set((a,b) if a<=b else (b,a) for a,b in clusters)
        
        conts_in_clusters = set()
        for m in clusters:
            conts_in_clusters = conts_in_clusters | set(m)
    
        uniq_conts = len(conts_in_clusters)    
        print 'Unique contributors : ', uniq_conts
        
        # global blacklist - this dictionary holds each contributor's global blacklist (i.e. the one generated from his cluster)
        gub_blacklists = dict()
        gub_whitelists = dict()
        
        # intersection blacklist - this dictionary holds each contributor's intersection blacklist (i.e. the ips on his training set intersected 
        # with the blacklists of the contributors in his cluster)
        int_blacklists = dict()
        int_whitelists = dict()
                
        # what happens in the cluster stays in the cluster     
        for subset in clusters:
                
            criterion = train_set.target_ip.map(lambda x: x in subset)
            logs = train_set[criterion].copy()
                
            # get the cluster's contributors
            c_contributors = [x for x in subset]

            # compute the gub set for the cluster
            gub_bl = set()
            gub_wl = set()
            gub_bl, gub_wl = gub_prediction(c_contributors, l_blacklists, l_whitelists)
        
            for contributor in c_contributors:
        
                gub_blacklists[contributor] = gub_bl
                gub_whitelists[contributor] = gub_wl  
    
                # compute intersection blacklists
                int_bl_set = set()
                int_wl_set = set()
                int_bl_set, int_wl_set = intersection_prediction(contributor, c_contributors, l_blacklists, l_whitelists, victim_set)
                int_blacklists[contributor] = int_bl_set
                int_whitelists[contributor] = int_wl_set
                del int_bl_set; del int_wl_set
            
            del gub_bl; del gub_wl
                                            
        # predictions verification part
        for target in top_targets:
            if target in conts_in_clusters:
                stats = verify_prediction(l_blacklists[target], l_whitelists[target], gub_blacklists[target], gub_whitelists[target], int_blacklists[target], int_whitelists[target], set( test_set[ (test_set.target_ip == target) ].src_ip ) )
                stats["k"] = perc
                stats["n_clusters"] = pairs
                stats["unique_conts"] = uniq_conts
            else:
                #no sharing contribs
                stats = verify_prediction(l_blacklists[target], l_whitelists[target], l_blacklists[target], l_whitelists[target], l_blacklists[target], l_whitelists[target], set( test_set[ (test_set.target_ip == target) ].src_ip ) )
                stats["n_clusters"] = 0
                stats["k"] = 0
                
            stats["D"] = last_day
            stats["target"] = target

            stats_list.append(stats)    

        del gub_blacklists; del int_blacklists; del gub_whitelists; del int_whitelists
        
    del l_blacklists; l_whitelists
    del train_set; del test_set; 
    
df_stats = pd.DataFrame(stats_list)

# save the df for later processing
df_stats.to_pickle("../results/dimva_global_stats.pkl")