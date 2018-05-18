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

def compute_tp_ca_impr(x):
    if x['tp_local'] > 0:
        return (x['tp_ca'] - x['tp_local']) / float(x['tp_local'])
    else:
        return 0.0
        
def compute_tp_knn_ca_impr(x):
    if x['tp_local'] > 0:
        return (x['tp_knn_ca'] - x['tp_local']) / float(x['tp_local'])
    else:
        return 0.0        

def compute_fn_ca_impr(x):
    if x['fn_local'] > 0:
        return (x['fn_ca'] - x['fn_local']) / float(x['fn_local'])
    else:
        return 0.0

def compute_fn_knn_ca_impr(x):
    if x['fn_local'] > 0:
        return (x['fn_knn_ca'] - x['fn_local']) / float(x['fn_local'])
    else:
        return 0.0

def compute_fp_ca_incr(x):
    if x['fp_local'] > 0:
        return (x['fp_ca'] - x['fp_local']) / float(x['fp_local'])
    else:
        return 0.0
        
def compute_fp_knn_ca_incr(x):
    if x['fp_local'] > 0:
        return (x['fp_knn_ca'] - x['fp_local']) / float(x['fp_local'])
    else:
        return 0.0        
        
def plot_tpr(method, stats):
    
    k_values = []
    local_tpr = []; ca_tpr = []; ca_knn_tpr = [];
    
    for k in np.unique(stats["k"]):
        
        k_stats = stats[stats.k == k]
        k_values.append(k)
        local_tpr.append(k_stats['tp_local'].sum() / float(k_stats['tp_local'].sum() + k_stats['fn_local'].sum())) 
        ca_tpr.append(k_stats['tp_ca'].sum() / float(k_stats['tp_ca'].sum() + k_stats['fn_ca'].sum()))
        ca_knn_tpr.append(k_stats['tp_knn_ca'].sum() / float(k_stats['tp_knn_ca'].sum() + k_stats['fn_knn_ca'].sum()))
            
    plt.xlabel('k-NN' + ' k', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)

    plt.plot(k_values, local_tpr, color = colors[0], ls = linestyles[1], label='TS', marker = 'x', markersize = 5)    
    plt.plot(k_values, ca_tpr, color = colors[1], ls = linestyles[1], label='TS-CA', marker = 'o', markersize = 5)    
    plt.plot(k_values, ca_knn_tpr, color = colors[2], ls = linestyles[1], label='TS-CA-k-NN', marker = '^', markersize = 5)
    
    legend = plt.legend(loc='upper left', fontsize='small') 

    plt.savefig(method + '-TPR.pdf', bbox_inches='tight')
    plt.close()        

def plot_precision(method, stats):
    
    k_values = []
    local_pr = []; ca_pr = []; ca_knn_pr = []; 
    
    for k in np.unique(stats["k"]):
        
        k_stats = stats[stats.k == k]
        k_values.append(k)
        local_pr.append(k_stats['tp_local'].sum() / float(k_stats['tp_local'].sum() + k_stats['fp_local'].sum())) 
        ca_pr.append(k_stats['tp_ca'].sum() / float(k_stats['tp_ca'].sum() + k_stats['fp_ca'].sum()))
        ca_knn_pr.append(k_stats['tp_knn_ca'].sum() / float(k_stats['tp_knn_ca'].sum() + k_stats['fp_knn_ca'].sum()))
                    
    plt.xlabel('k-NN' + ' k', fontsize=14)
    plt.ylabel('Precision (PPV)', fontsize=14)

    plt.plot(k_values, local_pr, color = colors[0], ls = linestyles[1], label='TS', marker = 'x', markersize = 5)    
    plt.plot(k_values, ca_pr, color = colors[1], ls = linestyles[1], label='TS-CA', marker = 'o', markersize = 5)    
    plt.plot(k_values, ca_knn_pr, color = colors[2], ls = linestyles[1], label='TS-CA-k-NN', marker = '^', markersize = 5)

    legend = plt.legend(loc='lower left', fontsize='small') 

    plt.savefig(method + '-PPV.pdf', bbox_inches='tight')
    plt.close()
                
def plot_tp_improvement(method, stats):
    
    k_values = []
    ca_tp_impr = []; ca_knn_tp_impr = []
    
    for k in np.unique(stats["k"]):
        
        k_values.append(k)
        k_stats = stats[stats.k == k]
        
        ca_tp_impr.append(k_stats['tp_ca_impr'].mean())
        ca_knn_tp_impr.append(k_stats['tp_knn_ca_impr'].mean())        
            
    plt.xlabel('k-NN' + ' k', fontsize=14)
    plt.ylabel('Average Improvement of True Positives (TP)', fontsize=14)
        
    plt.plot(k_values, ca_tp_impr, color = colors[1], ls = linestyles[1], label='TS-CA', marker = 'o', markersize = 5)
    plt.plot(k_values, ca_knn_tp_impr, color = colors[2], ls = linestyles[1], label='TS-CA-k-NN', marker = '^', markersize = 5)

    legend = plt.legend(loc='upper left', fontsize='small') 

    plt.savefig(method + '-tp-impr.pdf', bbox_inches='tight')
    plt.close()                    

def plot_fp_increase(method, stats):
    
    k_values = []
    ca_fp_impr = []; ca_knn_fp_impr = []
    
    for k in np.unique(stats["k"]):
        
        k_values.append(k)
        k_stats = stats[stats.k == k]
        ca_fp_impr.append(k_stats['fp_ca_incr'].mean())
        ca_knn_fp_impr.append(k_stats['fp_knn_ca_incr'].mean())
            
    plt.xlabel('k-NN' + ' k', fontsize=14)
    plt.ylabel('Average Increase of False Positives (FP) - Log Scale', fontsize=14)
    plt.yscale('log')
    
    plt.plot(k_values, ca_fp_impr, color = colors[1], ls = linestyles[1], label='TS-CA', marker = 'o', markersize = 5)    
    plt.plot(k_values, ca_knn_fp_impr, color = colors[2], ls = linestyles[1], label='TS-CA-k-NN', marker = '^', markersize = 5)
    
    legend = plt.legend(loc='upper left', fontsize='small') 

    plt.savefig(method + '-fp-incr.pdf', bbox_inches='tight')    
    plt.close()

def plot_fn_increase(method, stats):
    
    k_values = []
    ca_fn_impr = []; ca_knn_fn_impr = []
    
    for k in np.unique(stats["k"]):
        
        k_values.append(k)
        k_stats = stats[stats.k == k]
        ca_fn_impr.append(k_stats['fn_ca_incr'].mean())
        ca_knn_fn_impr.append(k_stats['fn_knn_ca_incr'].mean())
            
    plt.xlabel('k-NN' + ' k', fontsize=14)
    plt.ylabel('Average Increase of False Negatives (FN)', fontsize=14)

    plt.plot(k_values, ca_fn_impr, color = colors[1], ls = linestyles[1], label='TS-CA', marker = 'o', markersize = 5)    
    plt.plot(k_values, ca_knn_fn_impr, color = colors[2], ls = linestyles[1], label='TS-CA-k-NN', marker = '^', markersize = 5)
    
    legend = plt.legend(loc='center right', fontsize='small') 

    plt.savefig(method + '-fn-incr.pdf', bbox_inches='tight')    
    plt.close()

def plot_f1_measure(method, stats):
    
    k_values = []
    
    local_tpr = []; ca_tpr = []; ca_knn_tpr = [];
    local_pr = []; ca_pr = []; ca_knn_pr = []; 
    
    for k in np.unique(stats["k"]):
        
        k_stats = stats[stats.k == k]
        k_values.append(k)
        
        local_pr.append(k_stats['tp_local'].sum() / float(k_stats['tp_local'].sum() + k_stats['fp_local'].sum())) 
        ca_pr.append(k_stats['tp_ca'].sum() / float(k_stats['tp_ca'].sum() + k_stats['fp_ca'].sum()))
        ca_knn_pr.append(k_stats['tp_knn_ca'].sum() / float(k_stats['tp_knn_ca'].sum() + k_stats['fp_knn_ca'].sum()))
            
        local_tpr.append(k_stats['tp_local'].sum() / float(k_stats['tp_local'].sum() + k_stats['fn_local'].sum())) 
        ca_tpr.append(k_stats['tp_ca'].sum() / float(k_stats['tp_ca'].sum() + k_stats['fn_ca'].sum()))
        ca_knn_tpr.append(k_stats['tp_knn_ca'].sum() / float(k_stats['tp_knn_ca'].sum() + k_stats['fn_knn_ca'].sum()))
            
    local_f2= [2 * a * b / float(a + b) for a, b in zip(local_pr, local_tpr)]
    ca_f2= [2 * a * b / float(a + b) for a, b in zip(ca_pr, ca_tpr)]
    ca_knn_f2= [2 * a * b / float(a + b) for a, b in zip(ca_knn_pr, ca_knn_tpr)]
    
    plt.xlabel('k-NN' + ' k', fontsize=14)
    plt.ylabel('F1 Measure', fontsize=14)
    
    plt.plot(k_values, local_f2, color = colors[0], ls = linestyles[1], label='TS', marker = 'x', markersize = 8, linewidth=1.0)    
    plt.plot(k_values, ca_f2, color = colors[1], ls = linestyles[1], label='TS-CA', marker = 'o', markersize = 8, linewidth=1.0)    
    plt.plot(k_values, ca_knn_f2, color = colors[2], ls = linestyles[1], label='TS-CA-k-NN', marker = '^', markersize = 8, linewidth=1.0)
        
    legend = plt.legend(numpoints=1, loc='lower left', fontsize='small') 
    
    plt.savefig(method + '-F1.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
              
    method = 'soldo'
    
    df = pd.read_pickle('soldo_stats_1.pkl')
    
    for i in [5, 10, 15, 20, 25, 30, 35]:
    
        print 'Loading pickle ', i
        temp_data = pd.read_pickle('soldo_stats_' + str(i) + '.pkl')
    
        df = df.append(temp_data)
    
    df['tp_ca_impr'] = df.apply(compute_tp_ca_impr, axis=1)
    df['tp_knn_ca_impr'] = df.apply(compute_tp_knn_ca_impr, axis=1)
    df['fp_ca_incr'] = df.apply(compute_fp_ca_incr, axis=1)
    df['fp_knn_ca_incr'] = df.apply(compute_fp_knn_ca_incr, axis=1)
    df['fn_ca_incr'] = df.apply(compute_fn_ca_impr, axis=1)
    df['fn_knn_ca_incr'] = df.apply(compute_fn_knn_ca_impr, axis=1)

    plot_tp_improvement(method, df)
    plot_fp_increase(method, df)
    plot_fn_increase(method, df)
    plot_tpr(method, df)
    plot_precision(method, df)
    plot_f1_measure(method, df)