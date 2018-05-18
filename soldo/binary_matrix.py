# create the binary matrix
def compute_binary_matrix(victim_set, logs, training_window, offset, window):
    
    # the binary matrix with keys of the form 'victim' contains a second dictionary with keys 
    # 'src_ip' and values the binary list for all days e.g. [1, 0, 0, 0, 0]
    binary_matrix = dict()
    
    for victim in victim_set:
        binary_matrix[victim] = dict()              
    
    c = logs.groupby(['target_ip', 'src_ip'])
    
    # iterate through the groups
    for group, row in c:
        
        binary_matrix[group[0]][group[1]] = [0 for r in xrange(training_window)]
        
        # get the days of attacks
        d = row['D']
        for day in d:
            binary_matrix[group[0]][group[1]][day.day - window - offset] = 1            
        del d
    del c
                    
    # write a file with the binary matrix
    file = open('stats/stats' + str(window) + '/binarymatrix.txt', 'w')
    
    for victim in binary_matrix:
        file.write('Victim :' + str(victim) + '\n')
        for attacker in binary_matrix[victim]:    
            file.write('Attacker : ' + str(attacker) + ' BM : ' + str(binary_matrix[victim][attacker]) + '\n')
    file.close()
    
    return binary_matrix