# compute the combined score for every victim based on TS scores and CA scores
def compute_ts_ca_score(ts_scores, ca_scores, ca_densities, avg_ca_strength, window):
    
    # dictionary for storing the combined scores for each victim
    combined_scores = dict()

    # for every victim
    for victim in ts_scores:
        
        combined_scores[victim] = dict()
        
        attacker_set = set(ts_scores[victim].keys())
        attacker_set.update(ca_scores[victim].keys())
                
        # for every attacker
        for attacker in attacker_set:
            
            try:
                # get the ts score for this attacker
                ts = ts_scores[victim][attacker]
            except KeyError:
                ts = 0.0
            try:
                # get the ca score 
                cas = ca_scores[victim][attacker]
                # get the weight for this victim/attacker pair                
                w_cas = compute_ca_weight(ca_densities[victim][attacker], avg_ca_strength) * cas
            except KeyError:
                w_cas = 0.0
                
            # sum the scores and store them to combined_scores dict
            combined_scores[victim][attacker] = ts + w_cas
     
    # store the combined scores to a file
    fi = open('stats/stats' + str(window) + '/ts_ca_prediction_scores.txt', 'w')
    for victim in combined_scores:
        fi.write('Victim : ' + str(victim) + '\n')
        for attacker in combined_scores[victim]:
            fi.write('Attacker : ' + str(attacker) + ' || Prediction : ' + str(combined_scores[victim][attacker]) +'\n')                
    
    fi.close()
    
    return combined_scores

# compute the combined score for every victim based on TS scores, kNN scores and CA scores
def compute_ts_knn_ca_score(ts_scores, nn_scores, ca_scores, strength, avg_nn_strength, ca_densities, avg_ca_strength, window):
    
    # dictionary for storing the combined scores for each victim
    combined_scores = dict()

    # for every victim
    for victim in ts_scores:
        
        combined_scores[victim] = dict()
        
        # compute the nn_weight for this victim (1 if he has strong neighborhood or 0 otherwise)
        nn_weight = compute_nn_weight(strength[victim], avg_nn_strength)
        
        attacker_set = set(ts_scores[victim].keys())
        attacker_set.update(ca_scores[victim].keys())
        attacker_set.update(nn_scores[victim].keys())
                    
        # for every attacker
        for attacker in attacker_set:
            
            try:
                # get the ts score for this attacker
                ts = ts_scores[victim][attacker]
            except KeyError:
                ts = 0.0
            try:
                # get the nearest neighbors score
                nns = nn_scores[victim][attacker]
                w_nns = nn_weight * nns
            except KeyError:
                w_nns = 0.0    
            try:
                # get the ca score 
                cas = ca_scores[victim][attacker]
                # get the weight for this victim/attacker pair                
                w_cas = compute_ca_weight(ca_densities[victim][attacker], avg_ca_strength) * cas
            except KeyError:
                w_cas = 0.0    
                
            # sum the scores and store them to combined_scores dict
            combined_scores[victim][attacker] = ts + w_nns + w_cas
     
    # store the combined scores to a file
    fi = open('stats/stats' + str(window) + '/ts_knn_ca_prediction_scores.txt', 'w')
    for victim in combined_scores:
        fi.write('Victim : ' + str(victim) + '\n')
        for attacker in combined_scores[victim]:
            fi.write('Attacker : ' + str(attacker) + ' || Prediction : ' + str(combined_scores[victim][attacker]) +'\n')                
    
    fi.close()
            
    return combined_scores        

# compute the nn weight according to the formula sigma(s_uv) / (sigma(s_uv) + lambda1)
def compute_nn_weight(strength, avg_nn_strength):
    
    # the weight to return
    weight = 0
    
    # the lambda1 parameter configures the weight to knn
    lambda1 = 0
            
    # if we have a strong neighborhood the weight should be close to 1 else close to 0
    if ( strength > avg_nn_strength ):
        lambda1 = 10 ** (-6)
    else:
        lambda1 = 10 ** 6    
    
    # compute the weight 
    weight = strength / (strength + lambda1)
        
    return weight

# compute the ca weight according to the formula sigma(r_uv)
def compute_ca_weight(density, avg_density):
    
    # the weight to return
    weight = 0.0
    
    # the lambda1 parameter configures the weight to knn
    lambda2 = 0.0
            
    # if we have a dense neighborhood the weight should be close to 1 else close to 0
    if ( density > avg_density ):
        lambda2 = 10 ** (-6)
    else:
        lambda2 = 10 ** 6    
    
    # compute the weight 
    weight = density / (density + lambda2)
        
    return weight

# compute the average combined score in order to set the threshold
def compute_prediction_threshold_avg(combined_scores):
    
    thres = list()
    
    # for every victim
    for victim in combined_scores:
        # for every attacker
        for attacker in combined_scores[victim]:
            thres.append(combined_scores[victim][attacker])
    
    threshold = sum(thres) / len(thres)
    
    return threshold    
    
# compute prediction threshold based on the average of max and min combined values    
def compute_prediction_threshold_maxmin(combined_scores):
    
    thres = list()
    
    # for every victim
    for victim in combined_scores:
        # for every attacker
        for attacker in combined_scores[victim]:
            thres.append(combined_scores[victim][attacker])
    
    threshold = float(max(thres) - min(thres)) / 2 
    
    return threshold

# compute the time series threshold
def compute_prediction_threshold_ts(ts_scores):
    
    thres = list()
    
    # for every victim
    for victim in ts_scores:
        # for every attacker
        for attacker in ts_scores[victim]:
            thres.append(ts_scores[victim][attacker])
    
    threshold = float(max(thres) - min(thres)) / 2 
    
    return threshold

# generate the blacklist for every victim
def generate_blacklist(combined_scores, threshold, window):
    
    # a dictionary for storing the blacklist for each victim
    blacklists = dict()
    
    # for every victim
    for victim in combined_scores:
        
        blacklists[victim] = dict()        
            
        # for every attacker 
        for attacker in combined_scores[victim]:

            blacklists[victim][attacker] = 0
              
            # if the prediction exceeds the prediction threshold
            if combined_scores[victim][attacker] >= threshold:
                blacklists[victim][attacker] = 1
    
    # store the blacklists to a file
    fi = open('stats/stats' + str(window) + '/blacklists.txt', 'w')
    for victim in blacklists:
        fi.write('Victim : ' + str(victim) + '\n')
        for attacker in blacklists[victim]:
            fi.write('Attacker : ' + str(attacker) + ' || Prediction : ' + str(blacklists[victim][attacker]) +'\n')                
    
    fi.close()
    
    return blacklists
        
# compute the most likely attackers    
def compute_most_likely_attackers(preds, b_size, threshold):
    
    # list for storing the top score attackers
    top_score_attackers = []
    
    # convert dictionary to list of tuples
    l = [(k, v) for k, v in preds.iteritems() if v >= threshold]
    # sort the list of tuples according to score
    l.sort(key = lambda tup: tup[1], reverse = True)
    # keep the number of attackers that we want
    l = l[0:b_size]
    
    # assing the attackers to a new list
    top_score_attackers = [w[0] for w in l]
    
    return top_score_attackers
    
# compute the least likely attackers    
def compute_least_likely_attackers(preds, w_size, threshold):
    
    # list for storing the least score attackers
    least_score_attackers = []
    
    # convert dictionary to list of tuples
    l = [(k, v) for k, v in preds.iteritems() if v < threshold]
    # sort the list of tuples according to score
    l.sort(key = lambda tup: tup[1])
    # keep the number of attackers that we want
    l = l[0:w_size]
    
    # assing the attackers to a new list
    least_score_attackers = [w[0] for w in l]
    
    return least_score_attackers