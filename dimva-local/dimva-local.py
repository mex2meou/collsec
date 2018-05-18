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

from sklearn.neighbors import NearestNeighbors

stats_list = []
total_pairs = (70 * 69) / 2
pairs = [1, 5, 10, 15, 20, 25, 30, 35]

kNN_alg = ['auto', 'ball_tree', 'kd_tree', 'brute']
    
for i in range(0, num_tests):
    
    print 'Window: ', i
    start_day = logs_start_day + dt.timedelta(days=i)
    
    # load the window data into a dataframe
    
    # load the sample file for testing purposes
    # window_logs = pd.read_pickle(data_dir + "sample.pkl")
    
    # or the actual data for the experiments
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
    ind_orgs = dict( zip(top_targets, range(top_targets.size) ) )
    reverse_ind_orgs = dict( zip(ind_orgs.values(), ind_orgs.keys()) )
    
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
            o2o[ ind_orgs[pair[0]], ind_orgs[pair[1]], idx] = len( victim_daily_set[pair[0]][idx] & victim_daily_set[pair[1]][idx])
            o2o[ ind_orgs[pair[1]], ind_orgs[pair[0]], idx] = o2o[ind_orgs[pair[0]], ind_orgs[pair[1]], idx]

    # local prediction and blacklist generation part - this dictionary holds each contributor's local blacklist       
    print 'Computing local predictions...'
    l_blacklists = dict()
    l_whitelists = dict()
    l_blacklists, l_whitelists = ts.local_prediction(top_targets, train_set, i)      
    
    X = o2o.sum(axis=2)
    
    # compute nearest neighbors based on the o2o matrix
    neighbors = NearestNeighbors(n_neighbors = 36, algorithm = kNN_alg[1]).fit( X )
    _, indices = neighbors.kneighbors(X)
        
    org_neighbors = dict()
    for idx, x in enumerate(indices):
        org_neighbors[reverse_ind_orgs[idx]] = [reverse_ind_orgs[y] for idy, y in enumerate(x)]
    
    # remove the organization itself as knn returns it and make sure it is 5 nearest neighbors
    for org in org_neighbors:
        if org in org_neighbors[org]:
            org_neighbors[org].remove(org)
        
        if len(org_neighbors[org]) > 35:
            org_neighbors[org].pop()
                        
    # dimva clustering part 
    for target in top_targets:
        
        print 'Contributor : ', target
        clusters = [((target, ) + tuple(org_neighbors[target][:pair])) for pair in pairs]
            
        # what happens in the cluster stays in the cluster     
        for cluster_idx, subset in enumerate(clusters):
            
            print 'Cluster id: ', cluster_idx
                            
            criterion = train_set.target_ip.map(lambda x: x in subset)
            logs = train_set[criterion].copy()
            # get the cluster's contributors
            c_contributors = [x for x in subset]
            
            # compute the gub set for the cluster
            gub_bl = set()
            gub_wl = set()
            gub_bl, gub_wl = gub_prediction(c_contributors, l_blacklists, l_whitelists)
            
            # compute intersection blacklists
            int_bl_set = set()
            int_wl_set = set()
            int_bl_set, int_wl_set = intersection_prediction(target, c_contributors, l_blacklists, l_whitelists, victim_set)
                                            
            stats = verify_prediction(l_blacklists[target], l_whitelists[target], gub_bl, gub_wl, int_bl_set, int_wl_set, set( test_set[ (test_set.target_ip == target) ].src_ip ) )
            stats["target"] = target
            stats["pair_idx"] = pairs[cluster_idx]
            stats["D"] = last_day
            stats_list.append(stats)    
            
            del gub_bl; del gub_wl; del int_bl_set; del int_wl_set
                
    del l_blacklists; l_whitelists
    del train_set; del test_set; 
    
df_stats = pd.DataFrame(stats_list)

# save the df for later processing
df_stats.to_pickle("../results/dimva_local_stats.pkl")