#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '../utils/')

from util import *

import time_series as ts

import numpy as np
import pandas as pd

from itertools import combinations
from itertools import product

from scipy.sparse import lil_matrix

from sklearn.neighbors import NearestNeighbors

from scipy.stats import itemfreq

kNN_alg = ['auto', 'ball_tree', 'kd_tree', 'brute']
nn_orgs = [1, 5, 10, 15, 20, 25, 30, 35]
nn_ips = 50

stats_list = []

for i in range(0, num_tests):

    print 'Window: ', i
    start_day = logs_start_day + dt.timedelta(days=i)

    # load the window data into a dataframe
    
    # load the sample for testing purposes
    # window_logs = pd.read_pickle(data_dir + "sample.pkl")
    
    # or load the data for the actual experiments
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
    o2o = np.zeros((top_targets.size, top_targets.size, len(train_date_list)))

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
    print 'Creating o2o matrix...'
    for pair in target_pairs:
        for idx, day in enumerate(train_date_list):
            o2o[ ind_orgs[pair[0]], ind_orgs[pair[1]], idx] = len( victim_daily_set[pair[0]][idx] & victim_daily_set[pair[1]][idx])
            o2o[ ind_orgs[pair[1]], ind_orgs[pair[0]], idx] = o2o[ind_orgs[pair[0]], ind_orgs[pair[1]], idx]
                        
    X = o2o.sum(axis = 2)
                        
    # local prediction and blacklist generation part - this dictionary holds each contributor's local blacklist
    print 'Computing local predictions...'
    l_blacklists = dict()
    l_whitelists = dict()
    l_blacklists, l_whitelists = ts.local_prediction(top_targets, train_set, i)
    
    # clustering part - kNN on organisations    
    for k in nn_orgs:
    
        print 'Kvalue :', k
        
        # compute nearest neighbors based on the ip2ip matrix
        neighbors = NearestNeighbors(n_neighbors = k, algorithm = kNN_alg[1]).fit( X )
        distances, indices = neighbors.kneighbors(X)
        
        # dict storing the neighborhood strength of each contributor's neighborhood (as the sum of intersection with the nearest contributors)
        neigh_strength = dict()
        
        for idx, x in enumerate(indices):
            arr = X[idx]
            strength = 0.0
            for y in x:
                strength = strength + arr[y]
            neigh_strength[idx] = strength        
        
        clust_thres = 0.0
        for l, v in neigh_strength.iteritems():
            clust_thres = clust_thres + v
        
        clust_thres = float(clust_thres / len(neigh_strength))   
        
        # dictionary storing for each contributor a list with its nearest neighbors
        org_neighbors = dict()
        for idx, x in enumerate(indices):
            if neigh_strength[idx] >= clust_thres:
                distance_threshold = np.percentile(distances[idx], 40)
                org_neighbors[reverse_ind_orgs[idx]] = [reverse_ind_orgs[y] for idy, y in enumerate(x) if distances[idx][idy] <= distance_threshold]
        
        number_of_clusters = len(org_neighbors)
        print 'Number of clusters : ', number_of_clusters
        
        cluster_size = 0
        for neigh in org_neighbors:
            cluster_size += len(org_neighbors[neigh])
            
        avg_cluster_size = cluster_size / number_of_clusters    
        print 'Average Neighborhood Size : ', avg_cluster_size
                                                
        # global blacklist - this dictionary holds each contributor's global blacklist (i.e. the one generated from his cluster)
        gub_blacklists = dict()
        gub_whitelists = dict()
        
        # intersection blacklist - this dictionary holds each contributor's intersection blacklist (i.e. the ips on his training set intersected 
        # with the blacklists of the contributors in his cluster)
        int_blacklists = dict()
        int_whitelists = dict()

        # ip2ip corelation blacklist
        ip2ip_blacklists = dict()    
        ip2ip_whitelists = dict()
    
        # combined ip2ip and intersection blacklist
        int_ip2ip_blacklists = dict()
        int_ip2ip_whiteslits = dict()
    
        # what happens in the cluster stays in the cluster     
        for contributor in org_neighbors:
           
            # get the cluster's contributors
            c_contributors = org_neighbors[contributor] + [contributor]
    
            # create the ip2ip matrix for the cluster
            criterion = train_set.target_ip.map(lambda x: x in c_contributors)
            logs = train_set[criterion].copy()

            top_attackers = getHeavyHitters( logs["src_ip"] , 0.9)
            
            # limit the attacker to top1000
            top_attackers = top_attackers[:1000]
            print 'Top attacks : ', top_attackers.size            
            
            ind_ips = dict( zip(top_attackers, range(top_attackers.size) ) )
            reverse_ind_ips = dict( zip(ind_ips.values(), ind_ips.keys()) )

            criterion = logs.src_ip.map(lambda x: x in top_attackers)
            logs = logs[criterion]
            logs.src_ip = logs.src_ip.map(lambda x : ind_ips[x])

            df_gr = logs.groupby("D").apply(lambda x: np.bincount( x["src_ip"], minlength=top_attackers.size) )
            
            ip2ip = lil_matrix(np.zeros( top_attackers.size**2, dtype=np.uint32))

            print 'computing ip2ip matrix...'
            for l, v in df_gr.iteritems():
                ip2ip += np.array([np.uint32(min(f)) for f in product(v,v)])
            
            del df_gr
            
            # compute nearest neighbors based on the ip2ip matrix
            nbrs = NearestNeighbors(n_neighbors= min(nn_ips, top_attackers.size), algorithm= kNN_alg[1]).fit( ip2ip.toarray().reshape(top_attackers.size, top_attackers.size) )
            _, indic = nbrs.kneighbors(ip2ip.toarray().reshape(top_attackers.size, top_attackers.size))

            del ip2ip; del top_attackers
            
            # for each attacker ip store the k corelated ips
            corelated_ips = dict()

            for idx, x in enumerate(indic):
                corelated_ips[reverse_ind_ips[idx]] = [reverse_ind_ips[y] for y in x]
    
            # compute gub blacklist
            gub_bl = set()
            gub_wl = set()
            gub_bl, gub_wl = gub_prediction(c_contributors, l_blacklists, l_whitelists)        
            
            gub_blacklists[contributor] = gub_bl
            gub_whitelists[contributor] = gub_wl
            del gub_bl; del gub_wl
            
            # compute intersection blacklists
            int_bl_set = set()
            int_wl_set = set()
            int_bl_set, int_wl_set = intersection_prediction(contributor, c_contributors, l_blacklists, l_whitelists, victim_set)
            int_blacklists[contributor] = int_bl_set
            int_whitelists[contributor] = int_wl_set
            del int_bl_set; del int_wl_set
                                                            
            # make ip2ip corelation prediction
            ip2ip_bl_set = set()
            ip2ip_wl_set = set()
            ip2ip_bl_set, ip2ip_wl_set = ip2ip_prediction(contributor, l_blacklists, l_whitelists, corelated_ips)            
            ip2ip_blacklists[contributor] = ip2ip_bl_set
            ip2ip_whitelists[contributor] = ip2ip_wl_set
            del ip2ip_bl_set; del ip2ip_wl_set
            
            # make combined ip2ip and intersection prediction
            int_ip2ip_bl_set = set()
            int_ip2ip_wl_set = set()
            int_ip2ip_bl_set, int_ip2ip_wl_set = combined_int_ip2ip_prediction(int_blacklists[contributor], ip2ip_blacklists[contributor], int_whitelists[contributor], ip2ip_whitelists[contributor])
            int_ip2ip_blacklists[contributor] = int_ip2ip_bl_set
            int_ip2ip_whiteslits[contributor] = int_ip2ip_wl_set
            del int_ip2ip_bl_set; del int_ip2ip_wl_set
            
            del corelated_ips;
            
        # predictions verification part
        for target in org_neighbors:

            stats = verify_prediction(l_blacklists[target], l_whitelists[target], gub_blacklists[target], gub_whitelists[target], int_blacklists[target], int_whitelists[target], ip2ip_blacklists[target], ip2ip_whitelists[target], int_ip2ip_blacklists[target], int_ip2ip_whiteslits[target], set( test_set[ (test_set.target_ip == target) ].src_ip ) )
                        
            stats["D"] = last_day
            stats["n_clusters"] = k
            stats["n_strong_clusters"] = number_of_clusters
            stats["avg_cluster_size"] = avg_cluster_size
            stats["target"] = target

            stats_list.append(stats)    
        
        del gub_blacklists; del int_blacklists; del ip2ip_blacklists; del int_ip2ip_blacklists
        
    del train_set; del test_set; del l_blacklists

df_stats = pd.DataFrame(stats_list)

# save the df for later processing
df_stats.to_pickle("../results/knn_stats.pkl")