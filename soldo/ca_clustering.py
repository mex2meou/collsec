# Cross associations clustering
import numpy as np
import scipy.sparse
import sys
sys.path.append('CA_python')

import ca_utils as ca_util

def compute_density_matrix(victim_set, attacker_set, binary_matrix, window_len, window):
    
    # the density matrix with keys of the form 'victim' contains a second dictionary with keys 
    # 'src_ip' and values the density list for all days e.g. [0.5, 0.3, 0.0, 0.5, 0]
    density_matrix = dict()
    
    total_victims = len(victim_set)
    total_attackers = len(attacker_set)
    
    for victim in victim_set:
        density_matrix[victim] = dict()
    
    for victim in victim_set:
        for attacker in binary_matrix[victim]:    
            density_matrix[victim][attacker] = [0.0 for r in xrange(window_len)]                      
    
    index_victims = dict()
    index_attackers = dict()
    
    for k, l in enumerate(victim_set):
        index_victims[l] = k
    
    inv_index_victims = {v: w for w, v in index_victims.items()}
    
    for k, l in enumerate(attacker_set):
        index_attackers[l] = k
    
    inv_index_attackers = {v: w for w, v in index_attackers.items()}
    
    # for all days of the training window    
    for day in range(0, window_len):
        
        print 'CA clustering - day : ', day
        
        row_index = []
        column_index = []
        
        for victim in victim_set:
            for attacker in binary_matrix[victim]:
                if binary_matrix[victim][attacker][day] == 1:
                    row_index.append(index_victims[victim])
                    column_index.append(index_attackers[attacker])
        
        # provide the indexes to the clustering algorithm
        A = np.array(zip(row_index, column_index))
        densities = ca_util.get_densities(A, total_victims, total_attackers)
        
        print densities.shape
                                    
        for victim in index_victims:
            
            att_ids = densities[index_victims[victim]].nonzero()[1]
            for att_id in att_ids:
                if density_matrix.has_key(inv_index_attackers[att_id]):    
                    density_matrix[inv_index_victims[index_victims[victim]]][inv_index_attackers[att_id]][day] = densities[index_victims[victim]][att_id]
                else:
                    density_matrix[inv_index_victims[index_victims[victim]]][inv_index_attackers[att_id]] = [0.0 for r in xrange(window_len)]
                    density_matrix[inv_index_victims[index_victims[victim]]][inv_index_attackers[att_id]][day] = densities[index_victims[victim], att_id]                               
                                    
    # write a file with the binary matrix
    file = open('stats/stats' + str(window) + '/densitymatrix.txt', 'w')
    
    for victim in density_matrix:
        file.write('Victim :' + str(victim) + '\n')
        for attacker in density_matrix[victim]:    
            file.write('Attacker : ' + str(attacker) + ' DM : ' + str(density_matrix[victim][attacker]) + '\n')
    file.close()
    
    return density_matrix
    
# compute the neighborhood strength for each victim / attacker
def compute_density_strength(density_matrix):
    
    strength = dict()
    
    for victim in density_matrix:
        strength[victim] = dict()
    
    # for all victims compute the strength of their neighborhood (sum of similarities with the neighbors)
    for victim in density_matrix:
        for attacker in density_matrix[victim]:        
            strength[victim][attacker] = sum(density_matrix[victim][attacker]) 
        
    return strength
    
def compute_avg_density_strength(strength_dict):
    
    avg_strength = 0.0
    summ = 0.0
    
    for victim in strength_dict:
        for attacker in strength_dict[victim]:
            summ += strength_dict[victim][attacker]    
        
    avg_strength = summ / (len(strength_dict.keys()) * len(strength_dict.values()))
    
    return avg_strength