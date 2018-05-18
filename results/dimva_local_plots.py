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

def plot_tpr(method, stats):
    
    k_values = []
    local_tpr = []; int_tpr = []; 
    
    for k in np.unique(stats["pair_idx"]):
        k_stats = stats[stats.pair_idx == k]
        k_values.append(k)
        local_tpr.append(k_stats['tp_local'].sum() / float(k_stats['tp_local'].sum() + k_stats['fn_local'].sum()))
        int_tpr.append(k_stats['tp_int'].sum() / float(k_stats['tp_int'].sum() + k_stats['fn_int'].sum()))
            
    plt.xlabel('Number of Pairs', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)

    plt.plot(k_values, local_tpr, color = colors[0], ls = linestyles[1], label='Local', marker = 'x', markersize = 5)    
    plt.plot(k_values, int_tpr, color = colors[2], ls = linestyles[1], label='Intersection', marker = '^', markersize = 5)

    legend = plt.legend(loc='upper left', fontsize='small') 
    plt.xticks(k_values)
    plt.savefig(method + '-TPR.pdf', bbox_inches='tight')
    plt.close()        

def plot_precision(method, stats):
    
    k_values = []
    local_pr = []; int_pr = []; 
    
    for k in np.unique(stats["pair_idx"]):
        k_stats = stats[stats.pair_idx == k]
        k_values.append(k)
        local_pr.append(k_stats['tp_local'].sum() / float(k_stats['tp_local'].sum() + k_stats['fp_local'].sum())) 
        int_pr.append(k_stats['tp_int'].sum() / float(k_stats['tp_int'].sum() + k_stats['fp_int'].sum()))
            
    plt.xlabel('Number of Pairs', fontsize=14)
    plt.ylabel('Precision (PPV)', fontsize=14)

    plt.plot(k_values, local_pr, color = colors[0], ls = linestyles[1], label='Local', marker = 'x', markersize = 5)    
    plt.plot(k_values, int_pr, color = colors[2], ls = linestyles[1], label='Intersection', marker = '^', markersize = 5)

    legend = plt.legend(loc='upper right', fontsize='small') 
    plt.xticks(k_values)    
    plt.savefig(method + '-PPV.pdf', bbox_inches='tight')
    plt.close()

def plot_tp_improvement(method, stats):
    
    k_values = []
    int_tp_impr = []; 
    
    for k in np.unique(stats["pair_idx"]):
        k_values.append(k)
        k_stats = stats[stats.pair_idx == k]
        int_tp_impr.append(k_stats['tp_impr_int'].mean())
            
    plt.xlabel('Number of Pairs', fontsize=14)
    plt.ylabel('Average Improvement of True Positives (TP)', fontsize=14)
    
    plt.plot(k_values, int_tp_impr, color = colors[2], ls = linestyles[1], label='Intersection', marker = '^', markersize = 5)
    plt.xticks(k_values)    
    legend = plt.legend(loc='upper left', fontsize='small') 

    plt.savefig(method + '-tp-impr.pdf', bbox_inches='tight')    
    plt.close()

def plot_fp_increase(method, stats):
    
    k_values = []
    int_fp_impr = []; 
    
    for k in np.unique(stats["pair_idx"]):
        k_values.append(k)
        k_stats = stats[stats.pair_idx == k]
        int_fp_impr.append(k_stats['fp_incr_int'].mean())
    
    plt.xlabel('Number of Pairs', fontsize=14)
    plt.ylabel('Average Increase of False Positives (FP)', fontsize=14)

    plt.plot(k_values, int_fp_impr, color = colors[2], ls = linestyles[1], label='Intersection', marker = '^', markersize = 5)
    plt.xticks(k_values)    
    legend = plt.legend(loc='upper left', fontsize='small') 

    plt.savefig(method + '-fp-incr.pdf', bbox_inches='tight')    
    plt.close()

def plot_fn_increase(method, stats):
    
    k_values = []
    int_fn_impr = []; 
    
    for k in np.unique(stats["pair_idx"]):
        k_values.append(k)
        k_stats = stats[stats.pair_idx == k]
        
        int_fn_impr.append(k_stats['fn_incr_int'].mean())
    
    plt.xlabel('Number of Pairs', fontsize=14)
    plt.ylabel('Average Increase of False Negatives (FN)', fontsize=14)

    plt.plot(k_values, int_fn_impr, color = colors[2], ls = linestyles[1], label='Intersection', marker = '^', markersize = 5)
    plt.xticks(k_values)    
    legend = plt.legend(loc='center right', fontsize='small') 

    plt.savefig(method + '-fn-incr.pdf', bbox_inches='tight')    
    plt.close()

def plot_f1_measure(method, stats):
    
    k_values = []
    
    local_tpr = []; int_tpr = []; 
    local_pr = []; int_pr = []; 
    
    for k in np.unique(stats["pair_idx"]):
        k_stats = stats[stats.pair_idx == k]
        k_values.append(k)
        local_pr.append(k_stats['tp_local'].sum() / float(k_stats['tp_local'].sum() + k_stats['fp_local'].sum())) 
        int_pr.append(k_stats['tp_int'].sum() / float(k_stats['tp_int'].sum() + k_stats['fp_int'].sum()))
        
        local_tpr.append(k_stats['tp_local'].sum() / float(k_stats['tp_local'].sum() + k_stats['fn_local'].sum())) 
        int_tpr.append(k_stats['tp_int'].sum() / float(k_stats['tp_int'].sum() + k_stats['fn_int'].sum()))
    
    local_f1= [2 * a * b / float(a + b) for a, b in zip(local_pr, local_tpr)]
    int_f1= [2 * a * b / float(a + b) for a, b in zip(int_pr, int_tpr)]
    
    plt.xlabel('Number of Pairs', fontsize=14)
    plt.ylabel('F1 Measure', fontsize=14)
    
    plt.plot(k_values, local_f1, color = colors[0], ls = linestyles[1], label='Local', marker = 'x', markersize = 8, linewidth=1.0)    
    plt.plot(k_values, int_f1, color = colors[2], ls = linestyles[1], label='Intersection', marker = '^', markersize = 8, linewidth=1.0)
    
    plt.xticks(k_values)    
    legend = plt.legend(numpoints=1, loc='upper right', fontsize='small') 
    
    plt.savefig(method + '-F1.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
              
    method = 'dimva_local'
    
    df = pd.read_pickle(method + '_stats.pkl')

    df['fn_incr_gub'] = df.apply(compute_fn_gub_impr, axis=1)
    df['fn_incr_int'] = df.apply(compute_fn_int_impr, axis=1)

    plot_tp_improvement(method, df)
    plot_fp_increase(method, df)
    plot_fn_increase(method, df)
    plot_precision(method, df)
    plot_tpr(method, df)
    plot_f1_measure(method, df)