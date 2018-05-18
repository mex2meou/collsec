import pandas as pd
import datetime as dt
import numpy as np
import sPickle

# my imports
import binary_matrix as bm
import ewma_time_series as ewma
import ca_clustering as ca
import similarities as sim
import top_neighbors as tn
import blacklist as bl
import verify_prediction as vp

# ewma degree of weight decrease (value closer to 1 give more weight to most recent trends)
a = 0.9

# time window length
len_window = 6
train_w_length = 5
test_w_length = 1

# time offset
offset = 17 

# start - end date
start = 17
end = 31

# number of windows - 10 for the experiments (1 for the testing sample file)
windows = 10

logs_start_day = dt.datetime.strptime("2015-05-17", '%Y-%m-%d')

data_dir = '../data/' # directory where the data are stored 
data_prefix = 'df_sample_'

overall_windows_stats = pd.DataFrame()

# set to True if the CA densities have been calculated in a previous experiment
ca_dict_computed = False

print 'Script Starting Time: ', dt.datetime.now().isoformat()

for i in range(0, windows):

    print 'Window : ', i    

    start_day = logs_start_day + dt.timedelta(days = i)

    # load the window data
    
    # the sample file for testing
    # window_logs = pd.read_pickle(data_dir + "sample.pkl")
    
    # or the actual data
    window_logs = pd.read_pickle(data_dir + data_prefix + start_day.date().isoformat() + ".pkl")
    
    # extract /24 subnets from IPs
    window_logs.src_ip = window_logs.src_ip.map(lambda x: x[:11])
    
    # get the days, as well as first day and last day
    days = np.unique(window_logs['D'])
    first_day, last_day = np.min(days), np.max(days)
    
    print 'First day:', first_day
    print 'Last day:', last_day
    
    # split training set and testing set
    train_date_list = [start_day.date() + dt.timedelta(days=x) for x in range(0, len_window - test_w_length)]
    train_set = window_logs[window_logs.D.isin(train_date_list)] 
    
    test_date_list = [start_day.date() + dt.timedelta(days=x) for x in range(train_w_length, len_window)]
    test_set = window_logs[window_logs.D.isin(test_date_list)]
    
    print 'Train dates: ', train_set['D'].min(), train_set['D'].max()
    print 'Test dates: ', test_set['D'].min(), test_set['D'].max()
    print 'Training set size: ', train_set.shape[0]
    print 'Test set size: ', test_set.shape[0]
    
    del window_logs
    
    # find unique victims (i.e. set V)
    uni_target_ips = train_set['target_ip'].unique()
    print 'Number of unique victims : ', len(uni_target_ips)

    # find unique attackers (i.e. set A)
    uni_attacker_ips = train_set['src_ip'].unique()
    print 'Number of unique attackers : ', len(uni_attacker_ips)
    
    # first create the binary matrix for victims/attackers
    print 'Computing the binary matrix for victim/attackers...'
    binary_data = dict()
    binary_data = bm.compute_binary_matrix(uni_target_ips, train_set, train_w_length, offset, i)

    # then run the CA algorithm 
    if ca_dict_computed == False:
        # compute the densities - i.e. run CA algorithm
        print 'Running CA clustering algorithm - computing density matrix...'
        ca_densities = dict()
        ca_densities = ca.compute_density_matrix(uni_target_ips, uni_attacker_ips, binary_data, train_w_length, i)        
        sPickle.s_dump( ca_densities.iteritems(), open( "densities" + str(i) + ".spkl", "w" ) )
        
    else:
        # load the computed density matrix for the window
        print 'Loading CA density matrix from file...'
        #ca_densities = dict()
        ca_densities = dict(sPickle.s_load( open( "densities" + str(i) + ".spkl") ))
        
    # compute the denominator needed for the similarities and store it in a dictionary
    print 'Computing the denominator for similarities...'
    sim_denom = dict()
    sim_denom = sim.compute_denominator(train_set, uni_target_ips, train_w_length, offset, i, start_day)

    # find similarities between victims
    print 'Computing the similarities between victims...'
    similarity = dict()
    similarity = sim.compute_similarities(uni_target_ips, train_w_length, binary_data, sim_denom, i)

    # compute the top k neigbhors of each victim based on the similarities
    print 'Computing top neighbors...'
    top_neigh = dict()
    top_neigh = tn.compute_top_neighbors(similarity, i)

    # compute the time series scores for each victim/attacker - i.e. local approach
    print 'Computing the time series scores...'
    ts_score = dict()
    ts_score = ewma.compute_ts_scores(binary_data, a, train_w_length, i)
    
    # compute the local threshold 
    ts_threshold = bl.compute_prediction_threshold_avg(ts_score)
    print 'TS threshold is: ', ts_threshold
    
    # compute the local blacklist
    ts_blacklist = dict()
    ts_blacklist = bl.generate_blacklist(ts_score, ts_threshold, i)
            
    # compute the time series score for each victim attacker based on the density of the cluster they belong to
    print 'Computing EWMA on CA density scores...'
    ca_score = dict()
    ca_score = ewma.compute_ca_scores(ca_densities, a, train_w_length, i)
    
    print 'Computing CA density strength for each victim/attacker...'
    density_strength = dict()
    density_strength = ca.compute_density_strength(ca_densities)
    
    print 'Computing the average CA density strength...'
    ca_strength = ca.compute_avg_density_strength(density_strength)
    
    # compute the combined score (TS and CA score)
    print 'Computing the combined score (TS, CA) for each victim/attacker...'
    combined_ts_ca_score = dict()
    combined_ts_ca_score = bl.compute_ts_ca_score(ts_score, ca_score, density_strength, ca_strength, i)
    
    # compute the prediction threshold
    ts_ca_threshold = bl.compute_prediction_threshold_avg(combined_ts_ca_score)
    print 'Combined TS - CA prediction threshold is : ', ts_ca_threshold
    
    # generate the ts - CA blacklist
    print 'Generating a TS - CA blacklist for every victim...'
    ts_ca_blacklist = dict()
    ts_ca_blacklist = bl.generate_blacklist(combined_ts_ca_score, ts_ca_threshold, i)    
    
    print 'Computing attacker set after union with nearest neighbors...'
    attacker_set_union = dict()
    attacker_set_union = tn.compute_attacker_set_union(ts_score, top_neigh)
    
    print 'Computing neighborhood strength for each victim...'
    neigh_strength = dict()
    neigh_strength = tn.compute_neighborhood_strength(ts_score, top_neigh, similarity)
    
    print 'Computing the average neighborhood strength...'
    nn_strength = tn.compute_avg_nn_strength(neigh_strength)
    
    # now compute the nearest neighbors scores
    print 'Computing the nearest neighbors scores...'
    nn_score = dict()
    nn_score = tn.compute_nn_scores(ts_score, attacker_set_union, similarity, top_neigh, neigh_strength, nn_strength, i)
        
    # compute the combined score (TS, kNN and CA score)
    print 'Computing the combined score (TS - kNN - CA) for each victim/attacker...'
    combined_ts_knn_ca_score = dict()
    combined_ts_knn_ca_score = bl.compute_ts_knn_ca_score(ts_score, nn_score, ca_score, neigh_strength, nn_strength, density_strength, ca_strength, i)

    # compute the prediction threshold
    ts_knn_ca_threshold = bl.compute_prediction_threshold_avg(combined_ts_knn_ca_score)
    print 'Combined prediction threshold is : ', ts_knn_ca_threshold

    # generate the blacklist
    print 'Generating a blacklist for every victim...'
    ts_knn_ca_blacklist = dict()
    ts_knn_ca_blacklist = bl.generate_blacklist(combined_ts_knn_ca_score, ts_knn_ca_threshold, i)

    print 'Verifying the predictions...'
    window_stats = vp.verify_prediction(ts_blacklist, ts_ca_blacklist, ts_knn_ca_blacklist, test_set, i)
    overall_windows_stats = overall_windows_stats.append(window_stats)
        
    # remove the dicts and dfs
    del uni_target_ips; del uni_attacker_ips;
    del train_set; del test_set; del binary_data; del sim_denom; del similarity; del top_neigh;
    del ts_score; del attacker_set_union; del neigh_strength; del nn_score; 
    del ts_knn_ca_blacklist; del ca_score; del combined_ts_knn_ca_score; del combined_ts_ca_score;
    
# store overall experiment stats
overall_windows_stats.to_pickle('../results/soldo_stats_' + str(tn.top_neighbors) + '.pkl')
print 'Script End Time: ', dt.datetime.now().isoformat()