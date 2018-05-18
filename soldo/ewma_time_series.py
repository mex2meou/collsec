# ewma prediction - compute a time series score for each victim/attacker based on their history

#compute the weights
def gen_weights(a, window):
    ws = list()
    
    for i in range(window - 1, -1, -1):
        w = a * ((1-a)**i)
        ws.append(w)
    return ws

# sum the weighted values
def weighted(data, ws):
    wt = list()
    
    for i, x in enumerate(data):
        wt.append(x*ws[i])
    
    return sum(wt)

# compute a time series score
def score(a, window, data):
    
    ws = gen_weights(a, window)
    pred = weighted(data, ws)
    
    return pred
    
# compute the time series scores for every victim and attackers
def compute_ts_scores(binary_matrix, a, t_window, window):
    
    # dictionary for storing prediction results (i.e. blacklists)
    ts_scores = dict()
    
    # for all victims
    for victim in binary_matrix:
        
        ts_scores[victim] = dict()
        
        # for all attackers that have attacked the victim (i.e. they exist in the binary matrix)
        for attacker in binary_matrix[victim]:
            
            # make a predictions based on the binary matrix data
            ts_scores[victim][attacker] = score(a, t_window, binary_matrix[victim][attacker])

    # store the ts scores into file for debugging            
    f = open('stats/stats' + str(window) + '/ts_scores.txt', 'w')

    for victim in ts_scores:
        f.write('Victim : ' + str(victim) + '\n')
        for attacker in ts_scores[victim]:
            f.write('Attacker : ' + str(attacker) + ' || TS Score : ' + str(ts_scores[victim][attacker]) + '\n')
    f.close()
                
    return ts_scores
    
# compute the time series scores for every victim and attackers
def compute_ca_scores(density_matrix, a, t_window, window):
    
    # dictionary for storing prediction results (i.e. blacklists)
    ca_scores = dict()
    
    # for all victims
    for victim in density_matrix:
        
        ca_scores[victim] = dict()
        
        # for all attackers that the victim belongs to a cluster
        for attacker in density_matrix[victim]:
            
            # make a predictions based on the binary matrix data
            ca_scores[victim][attacker] = score(a, t_window, density_matrix[victim][attacker])

    # store the ts scores into file for debugging            
    f = open('stats/stats' + str(window) + '/ca_scores.txt', 'w')

    for victim in ca_scores:
        f.write('Victim : ' + str(victim) + '\n')
        for attacker in ca_scores[victim]:
            f.write('Attacker : ' + str(attacker) + ' || CA Score : ' + str(ca_scores[victim][attacker]) + '\n')
    f.close()
                
    return ca_scores