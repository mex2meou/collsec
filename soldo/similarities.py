import numpy as np
import datetime as dt

# compute the denominator for each victim, for each day - just once
def compute_denominator(logs, victims, window_len, offset, window, start_day):
    
    # dictionary storing the sqrt(# of attacks per day) for each victim
    denominator = dict()
    
    # for all victims
    for victim in victims:
        
        denominator[victim] = dict()
        
        # keep the logs of the specific victim
        c = logs[logs['target_ip'] == victim]
        
        # for all days
        for day in range(0, window_len):
            
            # count the number of attacks reported by victim on that day
            d = c[c['D'] == start_day.date() + dt.timedelta(days=day)]
            attacks = len(d)
            denominator[victim][day] = np.sqrt(attacks)
            
            del d
        
        del c
        
    return denominator    

# compute the exp(-|t2 -t1|) * sigma(b_a,u(t1) * b_a,v(t2)) / |b_u(t1)| * | b_u(t2)|
def compute_sim(attacker_set, binary_matrix, victim1, victim2, day1, day2, denom):
    
    # the summation
    summation = 0.0
    
    nominator = sum(binary_matrix[victim1][attacker][day1] * binary_matrix[victim2][attacker][day2] for attacker in attacker_set)
    denominator = denom[victim1][day1] * denom[victim2][day2]
    
    if( denominator != 0):
        return ( np.exp(-np.abs(day2 - day1)) * nominator ) / denominator   
    else:
        return 0.0    

# compute the similarity between two victims
def compute_similarity(attacker_set, binary_matrix, t_window, victim1, victim2, denom):

    similarity = 0.0
                
    similarity = sum(compute_sim(attacker_set, binary_matrix, victim1, victim2, i, j, denom) for i in range(0, t_window) for j in range(i, t_window))
    
    return similarity        

# compute similartiy for all pairs of victims
def compute_similarities(victim_list, t_window, binary_matrix, denom, window):
    
    # dictionary for storing similarites
    similarities = dict()
        
    # take every pair of victims
    tuples = [(victim1, victim2) for victim1 in victim_list for victim2 in victim_list if victim1 != victim2]
    
    # for every victim pair        
    for pair in tuples:
        
        if pair[0] not in similarities.keys():
            similarities[pair[0]] = dict()
        
        # get the intersection of attackers between the two victims
        attackers_v1 = set(binary_matrix[pair[0]].keys())
        attackers_v2 = set(binary_matrix[pair[1]].keys())
        attackers = attackers_v1 & attackers_v2
        attacker = list(attackers)
        
        similarities[pair[0]][pair[1]] = compute_similarity(attacker, binary_matrix, t_window, pair[0], pair[1], denom)
            
    # write a file
    fi = open('stats/stats' + str(window) + '/similarities.txt', 'w')
    for victim in similarities:
        fi.write('Victim1: ' + str(victim) + '\n') 
        for victim1 in similarities[victim]:
            fi.write('Victim2 : ' + str(victim1) + ' || Similarity : ' + str(similarities[victim][victim1]) + '\n')
        fi.write('\n')
            
    fi.close()
    
    return similarities