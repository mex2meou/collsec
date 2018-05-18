# compute the top neighbors for each victim

# number of top neighbors that we want to compute (i.e. k for k-NN)
top_neighbors = 5

# create a dictionary of the form : 'victim', [neighbor1, neighbor2, ...]
def compute_top_neighbors(similarities, window):
    
    # the dictionary to store the top neighbors for each victim
    top_n = dict()
    
    # for every victim in the similarities dictionary
    for victim in similarities:
        
        # for every victim we store a list with the top neighbors
        top_n[victim] = []
        
        # get tuples of the form (victim, similarity)
        l = [(k,v) for k, v in similarities[victim].iteritems()]
        l.sort(key = lambda tup:tup[1], reverse = True)
        
        # keep the number of neighbors that we want
        l = l[0:top_neighbors]
        top_n[victim] = [w[0] for w in l]
                                                
    fi = open('stats/stats' + str(window) + '/top_neighbors.txt', 'w')
    for key in top_n:
        fi.write('Victim : ' + str(key) + '\n') 
        fi.write('Top Neighbors : ' + str(top_n[key]) + '\n')
        fi.write('\n')
    fi.close()
                 
    return top_n
 
# compute the union of attackers according to victim neighborhood
def compute_attacker_set_union(ts_score, top_neighbors):
    
    attackers_union = dict()
    
    for victim in ts_score:
        
        attackers_union[victim] = []
        
        # get the union of attackers between the victim and his nearest neighbors
        attacker_set = set(ts_score[victim].keys())
        
        for neighbor in top_neighbors[victim]:
            attacker_set.update(ts_score[neighbor].keys())
        
        attackers_union[victim] = list(attacker_set)    
    
    return attackers_union    

# compute the intersection of attackers according to victim neighborhood
def compute_attacker_set_intersection(ts_score):
    
    attackers_intersection = dict()
    
    for victim in ts_score:
        
        attackers_intersection[victim] = []
        
        # define the attacker set as the attackers known by the victim
        attacker_set = set(ts_score[victim].keys())
        
        attackers_intersection[victim] = list(attacker_set)    
    
    return attackers_intersection    

# compute the neighborhood strength for each victim / attacker
def compute_neighborhood_strength(ts_score, top_neighbors, similarities):
    
    strength = dict()
    
    # for all victims compute the strength of their neighborhood (sum of similarities with the neighbors)
    for victim in ts_score:
                
        strength[victim] = sum(similarities[victim][neighbor] for neighbor in top_neighbors[victim])
        
    return strength
    
# compute the average strength of the neighborhoods
def compute_avg_nn_strength(strength):
    
    avg_strength = 0.0
    
    summ = sum(strength[victim] for victim in strength)    
        
    avg_strength = summ / len(strength.keys())
    
    return avg_strength

# compute the average of maxmin neighborhood strength
def compute_maxmin_nn_strength(strength):

    stren_list = [strength[victim] for victim in strength]
    maxminstren = float(max(stren_list) - min(stren_list)) / 2
    
    return maxminstren
    
# compute the scores from the nearest neighbors
def compute_nn_scores(ts_score, attacker_set, similarities, top_neighbors, strength, nn_strength, window):

    # a dictionary to store the nn_scores
    nn_scores = dict()
    
    # for every victim
    for victim in ts_score:
                
        # dictionary to store the nn_scores for every victim
        nn_scores[victim] = dict()
        
        # compute the nearest neighbor scores only if the victim has strong neighborhood
        if strength[victim] >= nn_strength:    
        
            # compute the denominator for the victim's neighborhood
            denominator = strength[victim]
                
            # for all attackers get the score from the nearest neighbors
            for attacker in attacker_set[victim]:    
                
                nominator = 0.0 
                
                for neighbor in top_neighbors[victim]:
                    try:
                        nominator += similarities[victim][neighbor] * ts_score[neighbor][attacker]
                    except KeyError:
                        nominator += 0.0
                                            
                nn_scores[victim][attacker] = nominator / denominator
    
            del denominator        
                
    # store the nn scores into file for debugging            
    f = open('stats/stats' + str(window) + '/nn_scores.txt', 'w')

    for victim in nn_scores:
        f.write('Victim : ' + str(victim) + '\n')
        for attacker in nn_scores[victim]:
            f.write('Attacker : ' + str(attacker) + ' || NN Score : ' + str(nn_scores[victim][attacker]) + '\n')
    f.close()
                
    return nn_scores