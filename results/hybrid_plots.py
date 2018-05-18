#!/usr/bin/python
# -*- coding: utf-8 -*-

# file with utility variables & functions

import pandas as pd
import numpy as np 
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

linestyles = ['_', '-', '--', ':', '-.']
colors = ('b', 'r', 'g', 'c', 'm', 'y', 'k')

def compute_fn_gub_impr(x):
    if x['fn_local'] > 0:
        return (x['fn_gub'] - x['fn_local']) / float(x['fn_local'])
    else:
        return 0.0

def compute_fn_int_impr(x):
    if x['fn_local'] > 0:
        return (x['fn_int'] - x['fn_local']) / float(x['fn_local'])
    else:
        return 0.0

def compute_fn_ip2ip_impr(x):
    if x['fn_local'] > 0:
        return (x['fn_ip2ip'] - x['fn_local']) / float(x['fn_local'])
    else:
        return 0.0

def compute_fn_int_ip2ip_impr(x):
    if x['fn_local'] > 0:
        return (x['fn_int_ip2ip'] - x['fn_local']) / float(x['fn_local'])
    else:
        return 0.0
    
def plot_tpr(method, stats):
    
    k_values = []
    local_tpr = []; gub_tpr = []; int_tpr = []; ip2ip_tpr = []; int_ip2ip_tpr = []
    
    for k in np.unique(stats["n_clusters"]):
        if k!=0:
            k_stats = stats[stats.n_clusters == k]
            k_values.append(k)
            local_tpr.append(k_stats['tp_local'].sum() / float(k_stats['tp_local'].sum() + k_stats['fn_local'].sum())) 
            gub_tpr.append(k_stats['tp_gub'].sum() / float(k_stats['tp_gub'].sum() + k_stats['fn_gub'].sum()))
            int_tpr.append(k_stats['tp_int'].sum() / float(k_stats['tp_int'].sum() + k_stats['fn_int'].sum()))
            ip2ip_tpr.append(k_stats['tp_ip2ip'].sum() / float(k_stats['tp_ip2ip'].sum() + k_stats['fn_ip2ip'].sum()))
            int_ip2ip_tpr.append(k_stats['tp_int_ip2ip'].sum() / float(k_stats['tp_int_ip2ip'].sum() + k_stats['fn_int_ip2ip'].sum()))
            
    plt.xlabel(method + ' k', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)

    plt.plot(k_values, local_tpr, color = colors[0], ls = linestyles[1], label='Local', marker = 'x', markersize = 8, linewidth=1.0)    
    plt.plot(k_values, gub_tpr, color = colors[1], ls = linestyles[1], label='Global', marker = 'o', markersize = 8, linewidth=1.0)    
    plt.plot(k_values, int_tpr, color = colors[2], ls = linestyles[1], label='Intersection', marker = '^', markersize = 8, linewidth=1.0)
    plt.plot(k_values, ip2ip_tpr, color = colors[3], ls = linestyles[1], label='Ip2Ip', marker = 'D', markersize = 8, linewidth=1.0)
    plt.plot(k_values, int_ip2ip_tpr, color = colors[4], ls=linestyles[1], label='Ip2Ip + Intersection', marker= 'v', markersize=8, linewidth=1.0)

    legend = plt.legend(loc='upper right', fontsize='small') 

    plt.savefig(method + '-TPR.pdf', bbox_inches='tight')
    plt.close()        

def plot_precision(method, stats):
    k_values = []
    local_pr = []; gub_pr = []; int_pr = []; ip2ip_pr = []; int_ip2ip_pr = []
    
    for k in np.unique(stats["n_clusters"]):
        if k!=0:
            k_stats = stats[stats.n_clusters == k]
            k_values.append(k)
            local_pr.append(k_stats['tp_local'].sum() / float(k_stats['tp_local'].sum() + k_stats['fp_local'].sum())) 
            gub_pr.append(k_stats['tp_gub'].sum() / float(k_stats['tp_gub'].sum() + k_stats['fp_gub'].sum()))
            int_pr.append(k_stats['tp_int'].sum() / float(k_stats['tp_int'].sum() + k_stats['fp_int'].sum()))
            ip2ip_pr.append(k_stats['tp_ip2ip'].sum() / float(k_stats['tp_ip2ip'].sum() + k_stats['fp_ip2ip'].sum()))
            int_ip2ip_pr.append(k_stats['tp_int_ip2ip'].sum() / float(k_stats['tp_int_ip2ip'].sum() + k_stats['fp_int_ip2ip'].sum()))
            
    plt.xlabel(method + ' k', fontsize=14)
    plt.ylabel('Precision (PPV)', fontsize=14)

    plt.plot(k_values, local_pr, color = colors[0], ls = linestyles[1], label='Local', marker = 'x', markersize = 8, linewidth=1.0)    
    plt.plot(k_values, gub_pr, color = colors[1], ls = linestyles[1], label='Global', marker = 'o', markersize = 8, linewidth=1.0)    
    plt.plot(k_values, int_pr, color = colors[2], ls = linestyles[1], label='Intersection', marker = '^', markersize = 8, linewidth=1.0)
    plt.plot(k_values, ip2ip_pr, color = colors[3], ls = linestyles[1], label='Ip2Ip', marker = 'D', markersize = 8, linewidth=1.0)
    plt.plot(k_values, int_ip2ip_pr, color = colors[4], ls=linestyles[1], label='Ip2Ip + Intersection', marker= 'v', markersize=8, linewidth=1.0)

    legend = plt.legend(loc='center right', fontsize='small') 

    plt.savefig(method + '-PPV.pdf', bbox_inches='tight')
    plt.close()
                
def plot_tp_improvement(method, stats):
    
    k_values = []
    gub_tp_impr = []; int_tp_impr = []; ip2ip_tp_impr = []; int_ip2ip_tp_impr = []
    gub_tp_impr_std = []; int_tp_impr_std = []; ip2ip_tp_impr_std = []; int_ip2ip_tp_impr_std = []
    
    for k in np.unique(stats["n_clusters"]):
        if k!= 0 :
            k_values.append(k)
            k_stats = stats[stats.n_clusters == k]
            gub_tp_impr.append(k_stats['tp_impr_gub'].mean())
            int_tp_impr.append(k_stats['tp_impr_int'].mean())
            ip2ip_tp_impr.append(k_stats['tp_impr_ip2ip'].mean())
            int_ip2ip_tp_impr.append(k_stats['tp_impr_int_ip2ip'].mean())
            
    plt.xlabel(method + ' k', fontsize=14)
    plt.ylabel('Average Improvement of True Positives (TP)', fontsize=14)
    
    plt.plot(k_values, gub_tp_impr, color = colors[1], ls = linestyles[1], label='Global', marker = 'o', markersize = 8, linewidth=1.0)
    plt.plot(k_values, int_tp_impr, color = colors[2], ls = linestyles[1], label='Intersection', marker = '^', markersize = 8, linewidth=1.0)
    plt.plot(k_values, ip2ip_tp_impr, color = colors[3], ls = linestyles[1], label='Ip2Ip', marker = 'D', markersize = 8, linewidth=1.0)
    plt.plot(k_values, int_ip2ip_tp_impr, color = colors[4], ls=linestyles[1], label='Ip2Ip + Intersection', marker= 'v', markersize=8, linewidth=1.0)

    legend = plt.legend(loc='upper right', fontsize='small') 

    plt.savefig(method + '-tp-impr.pdf', bbox_inches='tight')    
    plt.close()                    

def plot_fp_increase(method, stats):
    
    k_values = []
    gub_fp_impr = []; int_fp_impr = []; ip2ip_fp_impr = []; int_ip2ip_fp_impr = []
    
    for k in np.unique(stats["n_clusters"]):
        if k!= 0 :
            k_values.append(k)
            k_stats = stats[stats.n_clusters == k]
            gub_fp_impr.append(k_stats['fp_incr_gub'].mean())
            int_fp_impr.append(k_stats['fp_incr_int'].mean())
            ip2ip_fp_impr.append(k_stats['fp_incr_ip2ip'].mean())
            int_ip2ip_fp_impr.append(k_stats['fp_incr_int_ip2ip'].mean())
    
    plt.xlabel(method + ' k', fontsize=14)
    plt.ylabel('Average Increase of False Positives (FP) - Log Scale', fontsize=14)
    plt.yscale('log')
    
    plt.plot(k_values, gub_fp_impr, color = colors[1], ls = linestyles[1], label='Global', marker = 'o', markersize = 8, linewidth=1.0)    
    plt.plot(k_values, int_fp_impr, color = colors[2], ls = linestyles[1], label='Intersection', marker = '^', markersize = 8, linewidth=1.0)
    plt.plot(k_values, ip2ip_fp_impr, color = colors[3], ls = linestyles[1], label='Ip2Ip', marker = 'D', markersize = 8, linewidth=1.0)
    plt.plot(k_values, int_ip2ip_fp_impr, color = colors[4], ls=linestyles[1], label='Ip2Ip + Intersection', marker= 'v', markersize=8, linewidth=1.0)
    
    legend = plt.legend(loc='upper right', fontsize='small') 

    plt.savefig(method + '-fp-incr.pdf', bbox_inches='tight')    
    plt.close()
    
def plot_fn_increase(method, stats):
    
    k_values = []
    gub_fn_impr = []; int_fn_impr = []; ip2ip_fn_impr = []; int_ip2ip_fn_impr = []
    
    for k in np.unique(stats["n_clusters"]):
        if k!= 0 :
            k_values.append(k)
            k_stats = stats[stats.n_clusters == k]
            gub_fn_impr.append(k_stats['fn_incr_gub'].mean())
            int_fn_impr.append(k_stats['fn_incr_int'].mean())
            ip2ip_fn_impr.append(k_stats['fn_incr_ip2ip'].mean())
            int_ip2ip_fn_impr.append(k_stats['fn_incr_int_ip2ip'].mean())
    
    plt.xlabel(method + ' k', fontsize=14)
    plt.ylabel('Average Increase of False Negatives (FN)', fontsize=14)
    
    plt.plot(k_values, gub_fn_impr, color = colors[1], ls = linestyles[1], label='Global', marker = 'o', markersize = 8, linewidth=1.0)    
    plt.plot(k_values, int_fn_impr, color = colors[2], ls = linestyles[1], label='Intersection', marker = '^', markersize = 8, linewidth=1.0)
    plt.plot(k_values, ip2ip_fn_impr, color = colors[3], ls = linestyles[1], label='Ip2Ip', marker = 'D', markersize = 8, linewidth=1.0)
    plt.plot(k_values, int_ip2ip_fn_impr, color = colors[4], ls=linestyles[1], label='Ip2Ip + Intersection', marker= 'v', markersize=8, linewidth=1.0)
    
    legend = plt.legend(loc='upper right', fontsize='small') 

    plt.savefig(method + '-fn-incr.pdf', bbox_inches='tight')    
    plt.close()

def plot_f1_measure(method, stats):
    
    k_values = []
    
    local_tpr = []; gub_tpr = []; int_tpr = []; ip2ip_tpr = []; int_ip2ip_tpr = []    
    local_pr = []; gub_pr = []; int_pr = []; ip2ip_pr = []; int_ip2ip_pr = []
    
    for k in np.unique(stats["n_clusters"]):
        if k!=0:
            k_stats = stats[stats.n_clusters == k]
            k_values.append(k)
            local_pr.append(k_stats['tp_local'].sum() / float(k_stats['tp_local'].sum() + k_stats['fp_local'].sum())) 
            gub_pr.append(k_stats['tp_gub'].sum() / float(k_stats['tp_gub'].sum() + k_stats['fp_gub'].sum()))
            int_pr.append(k_stats['tp_int'].sum() / float(k_stats['tp_int'].sum() + k_stats['fp_int'].sum()))
            ip2ip_pr.append(k_stats['tp_ip2ip'].sum() / float(k_stats['tp_ip2ip'].sum() + k_stats['fp_ip2ip'].sum()))
            int_ip2ip_pr.append(k_stats['tp_int_ip2ip'].sum() / float(k_stats['tp_int_ip2ip'].sum() + k_stats['fp_int_ip2ip'].sum()))
            
            local_tpr.append(k_stats['tp_local'].sum() / float(k_stats['tp_local'].sum() + k_stats['fn_local'].sum())) 
            gub_tpr.append(k_stats['tp_gub'].sum() / float(k_stats['tp_gub'].sum() + k_stats['fn_gub'].sum()))
            int_tpr.append(k_stats['tp_int'].sum() / float(k_stats['tp_int'].sum() + k_stats['fn_int'].sum()))
            ip2ip_tpr.append(k_stats['tp_ip2ip'].sum() / float(k_stats['tp_ip2ip'].sum() + k_stats['fn_ip2ip'].sum()))
            int_ip2ip_tpr.append(k_stats['tp_int_ip2ip'].sum() / float(k_stats['tp_int_ip2ip'].sum() + k_stats['fn_int_ip2ip'].sum()))
    
    local_f1= [2 * a * b / float(a + b) for a, b in zip(local_pr, local_tpr)]
    gub_f1= [2 * a * b / float(a + b) for a, b in zip(gub_pr, gub_tpr)]
    int_f1= [2 * a * b / float(a + b) for a, b in zip(int_pr, int_tpr)]
    ip2ip_f1= [2 * a * b / float(a + b) for a, b in zip(ip2ip_pr, ip2ip_tpr)]
    int_ip2ip_f1= [2 * a * b / float(a + b) for a, b in zip(int_ip2ip_pr, int_ip2ip_tpr)]
    
    plt.xlabel(method + ' k', fontsize=14)
    plt.ylabel('F1 Measure', fontsize=14)
        
    plt.plot(k_values, local_f1, color = colors[0], ls = linestyles[1], label='Local', marker = 'x', markersize = 8, linewidth=1.0)    
    plt.plot(k_values, gub_f1, color = colors[1], ls = linestyles[1], label='Global', marker = 'o', markersize = 8, linewidth=1.0)    
    plt.plot(k_values, int_f1, color = colors[2], ls = linestyles[1], label='Intersection', marker = '^', markersize = 8, linewidth=1.0)
    plt.plot(k_values, ip2ip_f1, color = colors[3], ls = linestyles[1], label='Ip2Ip', marker = 'D', markersize = 8, linewidth=1.0)
    plt.plot(k_values, int_ip2ip_f1, color = colors[4], ls=linestyles[1], label='Ip2Ip + Intersection', marker= 'v', markersize=8, linewidth=1.0)
        
    legend = plt.legend(loc='center right', fontsize='small') 
    
    plt.savefig(method + '-F1.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':              

    methods = ['agglomerative', 'kmeans', 'knn']
    
    for method in methods:    
        
        # load the results for this method
        df = pd.read_pickle(method + "_stats.pkl")
        
        df['fn_incr_gub'] = df.apply(compute_fn_gub_impr, axis=1)
        df['fn_incr_int'] = df.apply(compute_fn_int_impr, axis=1)
        df['fn_incr_ip2ip'] = df.apply(compute_fn_ip2ip_impr, axis=1)
        df['fn_incr_int_ip2ip'] = df.apply(compute_fn_int_ip2ip_impr, axis=1)

        plot_tp_improvement(method, df)
        plot_fp_increase(method, df)
        plot_fn_increase(method, df)
        plot_tpr(method, df)
        plot_precision(method, df)
        plot_f1_measure(method, df)